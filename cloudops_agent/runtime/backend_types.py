# runtime/backend_types.py
"""
Framework-agnostic contract between the Cloud-OpsBench harness and whatever
diagnostic agent (single ReAct agent, multi-agent system, LangChain/LangGraph
graph, ...) is being benchmarked.

The harness only ever sees this file's types. It does not know or care how
`AgentBackend.solve_case` is implemented internally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Type

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Tool contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """
    One diagnostic tool, described in a way that is not tied to any specific
    agent framework.

    `invoke` is a plain callable: `invoke(**kwargs) -> str`. It already does
    argument filtering and observation tracing (see `TraceCollector.wrap`),
    so whoever consumes it does not need to reimplement either.
    """

    name: str
    description: str
    args_schema: Optional[Type[BaseModel]]
    invoke: Callable[..., str]


# ---------------------------------------------------------------------------
# Final answer contract
# ---------------------------------------------------------------------------

class Prediction(BaseModel):
    rank: int
    fault_object: str
    root_cause: str


class FinalDiagnosis(BaseModel):
    key_evidence_summary: str
    top_3_predictions: List[Prediction]


@dataclass(frozen=True)
class RootCauseSpec:
    code: str
    target_kind: str  # "NODE" | "APP" | "NAMESPACE"
    description: str


# ---------------------------------------------------------------------------
# Tool call tracing (shared by every backend, so no backend has to hand-roll
# its own step log to satisfy the evaluator)
# ---------------------------------------------------------------------------

class BudgetExceededError(RuntimeError):
    """Raised by a traced tool call once the case's tool-call budget is spent."""


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: Dict[str, Any]
    observation: str
    latency: float
    timestamp: float
    agent_id: Optional[str] = None


class TraceCollector:
    """
    Wraps tool callables so every invocation - regardless of which agent or
    framework triggered it - is recorded automatically: name, arguments,
    observation, latency, timestamp, and (optionally) which agent/role made
    the call. Also enforces the case's tool-call budget in one place.
    """

    def __init__(self, max_tool_calls: Optional[int] = None):
        self.max_tool_calls = max_tool_calls
        self.records: List[ToolCallRecord] = []

    def wrap(self, tool_name: str, invoke: Callable[..., str], agent_id: Optional[str] = None) -> Callable[..., str]:
        def traced(**kwargs: Any) -> str:
            if self.max_tool_calls is not None and len(self.records) >= self.max_tool_calls:
                raise BudgetExceededError(
                    f"Tool call budget exhausted ({self.max_tool_calls} calls)."
                )
            start = time.perf_counter()
            observation = invoke(**kwargs)
            if not isinstance(observation, str):
                observation = str(observation)
            self.records.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    arguments=dict(kwargs),
                    observation=observation,
                    latency=time.perf_counter() - start,
                    timestamp=time.time(),
                    agent_id=agent_id,
                )
            )
            return observation

        return traced

    def as_steps(self) -> List[Dict[str, Any]]:
        """Render collected tool calls into the legacy StepRecord-shaped step list."""
        return [
            {
                "step_id": idx,
                "agent_id": record.agent_id,
                "action_type": "tool",
                "action_name": record.tool_name,
                "action_input": record.arguments,
                "observation": record.observation,
                "error": None,
                "tool_latency": record.latency,
                "timestamp": record.timestamp,
            }
            for idx, record in enumerate(self.records, start=1)
        ]


# ---------------------------------------------------------------------------
# Task input contract (benchmark -> agent, handed over once per case)
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    case_id: str
    system: str
    namespace: str
    symptom: str

    tools: List[ToolSpec]
    max_tool_calls: int

    valid_root_causes: List[RootCauseSpec]
    valid_targets: Dict[str, List[str]]  # {"node": [...], "app": [...], "namespace": [...]}
    answer_schema: Type[BaseModel]  # FinalDiagnosis

    trace: TraceCollector

    # Pass-through bookkeeping (fault_category, model_name, ground-truth
    # result, run_started_at, ...) that backends do not need to interpret
    # but should fold into whatever they persist, so partial/interrupted
    # traces stay just as debuggable as before this abstraction existed.
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result contract (agent -> benchmark, handed back once per case)
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    finished: bool
    stop_reason: Optional[str]
    final_answer: Optional[FinalDiagnosis]

    # If a backend already produces a full legacy trace (steps + metadata,
    # like the existing single-agent runtime does), it can hand it over here
    # and the harness will persist it as-is. Otherwise leave None and the
    # harness assembles the trace file from TaskSpec.trace.as_steps().
    raw_trace: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentBackend(Protocol):
    """Anything with this method can be benchmarked, regardless of framework."""

    def solve_case(self, task: TaskSpec) -> CaseResult: ...
