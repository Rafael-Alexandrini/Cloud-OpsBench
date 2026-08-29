# runtime/task_spec_builder.py
"""
Builds a TaskSpec (the structured, framework-agnostic case description) from
the same raw ingredients run.py already produces today: the SimpleTool list
from create_k8s_tools(), and the case's metadata.

This is the single place that turns "prose prompt blocks" into data. Every
backend (react_single_agent, two_agent_demo, or a future LangChain/LangGraph
one) is built from the same TaskSpec, so none of them can silently drift
from the others or "forget" a tool/root-cause/target that a hand-copied
prompt might have missed.
"""

from __future__ import annotations

from typing import Any, List

from prompts.RCA_candidate import (
    SYSTEM_VALID_NAMESPACES,
    SYSTEM_VALID_SERVICES,
    VALID_NODES,
    parse_root_cause_taxonomy,
)
from tools.adapters import call_tool
from tools.definition import SimpleTool

from .backend_types import FinalDiagnosis, RootCauseSpec, TaskSpec, ToolSpec, TraceCollector


def build_task_spec(
    tools_list: List[SimpleTool],
    case_id: str,
    system: str,
    namespace: str,
    symptom: str,
    max_tool_calls: int,
    metadata: dict | None = None,
) -> TaskSpec:
    """
    Args:
        tools_list: output of tools.definition.create_k8s_tools(...) - the
            same object run.py already builds today.
        case_id, system, namespace, symptom: case identity/context.
        max_tool_calls: shared tool-call budget for the whole case,
            regardless of how many agents end up sharing it.
    """
    trace = TraceCollector(max_tool_calls=max_tool_calls)

    tool_specs = [
        ToolSpec(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            # Reuses the existing argument-filtering logic in
            # tools/adapters.py::call_tool so behavior matches the legacy
            # single-agent path exactly; TraceCollector.wrap adds tracing
            # and budget enforcement on top, transparently.
            invoke=trace.wrap(tool.name, lambda tool=tool, **kwargs: call_tool(tool, kwargs)),
        )
        for tool in tools_list
    ]

    system_key = system if system in SYSTEM_VALID_SERVICES else "train-ticket"
    valid_targets = {
        "node": list(VALID_NODES),
        "app": list(SYSTEM_VALID_SERVICES[system_key]),
        "namespace": list(SYSTEM_VALID_NAMESPACES[system_key]),
    }

    valid_root_causes = [
        RootCauseSpec(code=item["code"], target_kind=item["target"], description=item["description"])
        for item in parse_root_cause_taxonomy()
    ]

    return TaskSpec(
        case_id=case_id,
        system=system,
        namespace=namespace,
        symptom=symptom,
        tools=tool_specs,
        max_tool_calls=max_tool_calls,
        valid_root_causes=valid_root_causes,
        valid_targets=valid_targets,
        answer_schema=FinalDiagnosis,
        trace=trace,
        metadata=metadata or {},
    )
