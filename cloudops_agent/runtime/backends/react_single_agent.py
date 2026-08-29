# runtime/backends/react_single_agent.py
"""
The existing single-agent ReAct runtime (PromptBuilder + ModelRunner +
OutputParser + ToolExecutor + AgentRuntime), reembrulhado atras da interface
AgentBackend.solve_case(TaskSpec) -> CaseResult.

Nada do comportamento muda: mesmo prompt (agent_prompt / build_expected_output
importados sem alteracao), mesmo loop, mesmo arquivo de trace no formato que
evaluation.py ja consome. A unica diferenca e que a entrada agora vem de um
TaskSpec estruturado em vez de argumentos soltos.
"""

from __future__ import annotations

from typing import Any, Dict

from prompts.RCA_candidate import agent_prompt, build_expected_output
from runtime.agent_runtime import AgentRuntime
from runtime.backend_types import CaseResult, FinalDiagnosis, TaskSpec
from runtime.logger import TraceLogger
from runtime.model_runner import ModelRunner
from runtime.output_parser import OutputParser
from runtime.prompt_builder import PromptBuilder
from runtime.state import init_case_state
from runtime.tool_executor import ToolExecutor
from tools.definition import SimpleTool
from tools.registry import build_tool_registry, render_tools_description


class ReactSingleAgentBackend:
    """Baseline: um unico agente, loop ReAct linear, um tool call por passo."""

    def __init__(self, llm_conf: Dict[str, Any], trace_dir: str):
        self.llm_conf = llm_conf
        self.trace_dir = trace_dir

    def solve_case(self, task: TaskSpec) -> CaseResult:
        # ToolSpec.invoke ja e a versao rastreada/filtrada (task_spec_builder
        # a montou via TraceCollector.wrap(call_tool(...))). Aqui so
        # embrulhamos de volta no formato que ToolExecutor/call_tool esperam
        # (objeto com ._run), para reusar o runtime existente sem tocar nele.
        simple_tools = [
            SimpleTool(
                name=tool_spec.name,
                description=tool_spec.description,
                run_fn=tool_spec.invoke,
                args_schema=tool_spec.args_schema,
            )
            for tool_spec in task.tools
        ]
        tool_registry = build_tool_registry(simple_tools)
        tools_description = render_tools_description(tool_registry)

        prompt_builder = PromptBuilder(
            tools_description=tools_description,
            backstory_prompt=agent_prompt,
            expected_output=build_expected_output(task.system),
        )
        model_runner = ModelRunner(
            model_name=self.llm_conf["model"],
            provider="openai_compatible",
            api_base=self.llm_conf["api_base"],
            api_key=self.llm_conf["api_key"],
            temperature=self.llm_conf.get("temperature", 0.0),
            max_tokens=self.llm_conf.get("max_tokens", 1024),
            timeout=self.llm_conf.get("timeout"),
        )
        output_parser = OutputParser()
        tool_executor = ToolExecutor(tool_registry=tool_registry)
        trace_logger = TraceLogger(trace_dir=self.trace_dir)

        runtime = AgentRuntime(
            prompt_builder=prompt_builder,
            model_runner=model_runner,
            output_parser=output_parser,
            tool_executor=tool_executor,
            trace_logger=trace_logger,
        )

        full_question = (
            f"The Kubernetes environment in namespace `{task.namespace}` is experiencing a fault. "
            f"A high-level symptom has been reported: '{task.symptom}'. "
            f"Diagnose the root cause of this incident."
        )
        state = init_case_state(
            case_id=task.case_id,
            system_name=task.system,
            question=full_question,
            max_steps=task.max_tool_calls,
            metadata={
                "namespace": task.namespace,
                "query": task.symptom,
                **task.metadata,
            },
        )

        final_state = runtime.run_case(state)

        final_answer_obj = None
        if final_state.final_answer:
            try:
                final_answer_obj = FinalDiagnosis.model_validate_json(final_state.final_answer)
            except Exception:
                final_answer_obj = None

        return CaseResult(
            case_id=task.case_id,
            finished=final_state.finished,
            stop_reason=final_state.stop_reason,
            final_answer=final_answer_obj,
            # AgentRuntime/TraceLogger already produced a complete, correct
            # trace file (same shape evaluation.py already reads) - hand it
            # over as-is instead of asking the harness to reconstruct it.
            raw_trace=final_state.to_dict(),
            metadata={"architecture": "react_single_agent"},
        )
