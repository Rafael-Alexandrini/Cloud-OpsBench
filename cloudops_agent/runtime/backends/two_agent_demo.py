# runtime/backends/two_agent_demo.py
"""
Uma segunda arquitetura, estruturalmente diferente da single-agent, para
provar que o AgentBackend/TaskSpec realmente permite trocar de estrutura sem
tocar em run.py/evaluation.py.

Dois papeis, dois prompts curtos e independentes:

- Investigator: so chama ferramentas (via TaskSpec.tools[i].invoke, ja
  rastreado). Nao conhece a taxonomia de root causes - nao decide nada sobre
  a causa raiz, so junta evidencia ate decidir chamar "HANDOFF".
- Synthesizer: nao tem ferramentas. Le o log de evidencias reunido pelo
  Investigator + a taxonomia/lista de alvos (vindas de TaskSpec, nao de texto
  copiado) e produz o JSON final.

O trace de tool calls NAO precisa ser montado a mao aqui: como toda chamada
passa por TaskSpec.tools[i].invoke, o TraceCollector ja registra tudo -
o backend so precisa devolver o final_answer.
"""

from __future__ import annotations

from typing import Any, Dict, List

from runtime.backend_types import BudgetExceededError, CaseResult, FinalDiagnosis, TaskSpec
from runtime.model_runner import ModelRunner
from runtime.output_parser import OutputParser
from tools.registry import render_tools_description


def _render_taxonomy(task: TaskSpec) -> str:
    return "\n".join(
        f"- {rc.code} (target: {rc.target_kind}): {rc.description}" for rc in task.valid_root_causes
    )


def _render_targets(task: TaskSpec) -> str:
    return (
        f"- Nodes: {task.valid_targets.get('node', [])}\n"
        f"- Apps: {task.valid_targets.get('app', [])}\n"
        f"- Namespaces: {task.valid_targets.get('namespace', [])}"
    )


def _build_investigator_prompt(task: TaskSpec, tools_description: str, history_text: str, step: int) -> str:
    return f"""You are the INVESTIGATOR agent in a two-agent diagnostic team.
Your only job is to gather evidence with tools. You do NOT decide the root cause - another agent (the Synthesizer) will do that from your findings.

## Incident
Namespace: {task.namespace}
Symptom: {task.symptom}

## Available tools
{tools_description}

## Rules
- Call exactly one tool per step, using this exact format:
Thought: <brief reasoning>
Action: <tool_name>
Action Input: <json object, {{}} if the tool takes no parameters>
- Once you have enough evidence for another agent to diagnose the root cause, stop calling tools and instead output:
Thought: <why you believe you have enough evidence>
Action: HANDOFF
Action Input: {{}}
- Budget: step {step} of {task.max_tool_calls}.

## Evidence gathered so far
{history_text}
"""


def _build_synthesizer_prompt(task: TaskSpec, evidence_log: str) -> str:
    return f"""You are the SYNTHESIZER agent in a two-agent diagnostic team.
You have no tools. You only read evidence already collected by the Investigator agent and produce the final diagnosis.

## Incident
Namespace: {task.namespace}
Symptom: {task.symptom}

## Evidence collected by the Investigator agent
{evidence_log}

## Valid root causes
{_render_taxonomy(task)}

## Valid objects
{_render_targets(task)}

## Output
Respond with JSON ONLY (no text before or after), in exactly this shape:
{{
  "key_evidence_summary": "...",
  "top_3_predictions": [
    {{"rank": 1, "fault_object": "Kind/Name", "root_cause": "..."}},
    {{"rank": 2, "fault_object": "Kind/Name", "root_cause": "..."}},
    {{"rank": 3, "fault_object": "Kind/Name", "root_cause": "..."}}
  ]
}}
"""


class TwoAgentBackend:
    """Investigator (tool-calling) + Synthesizer (diagnosis-only). Shares one LLM endpoint for both roles."""

    def __init__(self, llm_conf: Dict[str, Any], trace_dir: str):
        self.model_runner = ModelRunner(
            model_name=llm_conf["model"],
            provider="openai_compatible",
            api_base=llm_conf["api_base"],
            api_key=llm_conf["api_key"],
            temperature=llm_conf.get("temperature", 0.0),
            max_tokens=llm_conf.get("max_tokens", 1024),
            timeout=llm_conf.get("timeout"),
        )
        self.output_parser = OutputParser()
        self.trace_dir = trace_dir  # unused directly: harness persists via TaskSpec.trace

    def solve_case(self, task: TaskSpec) -> CaseResult:
        tools_by_name = {tool.name: tool for tool in task.tools}
        tools_description = render_tools_description(tools_by_name)

        history_lines: List[str] = []
        stop_reason = "budget_exhausted"

        for step in range(1, task.max_tool_calls + 1):
            prompt = _build_investigator_prompt(
                task, tools_description, "\n".join(history_lines) or "None yet.", step
            )
            generation = self.model_runner.generate(prompt)
            parsed = self.output_parser.parse(generation["text"])

            if parsed["type"] != "tool":
                stop_reason = "investigator_invalid_output"
                break

            action_name = parsed["action_name"]
            if action_name == "HANDOFF":
                stop_reason = "handoff"
                break

            tool = tools_by_name.get(action_name)
            if tool is None:
                history_lines.append(f"[step {step}] ERROR: unknown tool '{action_name}'")
                continue

            try:
                observation = tool.invoke(**(parsed["action_input"] or {}))
            except BudgetExceededError:
                stop_reason = "budget_exhausted"
                break

            history_lines.append(
                f"[step {step}] {action_name}({parsed['action_input']}) ->\n{observation}"
            )
        else:
            stop_reason = "budget_exhausted"

        synth_prompt = _build_synthesizer_prompt(
            task, "\n".join(history_lines) or "No evidence was collected."
        )
        synth_generation = self.model_runner.generate(synth_prompt)
        synth_parsed = self.output_parser.parse(synth_generation["text"])

        final_answer_obj = None
        if synth_parsed["type"] == "finish":
            try:
                final_answer_obj = FinalDiagnosis.model_validate(synth_parsed["final_json"])
            except Exception:
                final_answer_obj = None

        return CaseResult(
            case_id=task.case_id,
            finished=final_answer_obj is not None,
            stop_reason="final_answer" if final_answer_obj is not None else stop_reason,
            final_answer=final_answer_obj,
            raw_trace=None,  # harness assembles this from task.trace.as_steps()
            metadata={"architecture": "two_agent_demo"},
        )
