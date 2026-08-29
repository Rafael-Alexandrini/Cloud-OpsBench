from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# -------------------------------------------------------------------
# Make project root importable
# -------------------------------------------------------------------
AGENT_ROOT = Path(__file__).resolve().parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

# -------------------------------------------------------------------
# Project-local imports
# -------------------------------------------------------------------
from prompts.config_utils import load_config
from tools.definition import create_k8s_tools

from runtime.backend_types import CaseResult, TaskSpec
from runtime.backends import BACKENDS
from runtime.task_spec_builder import build_task_spec

MAX_CASES_TO_RUN = 50
DEFAULT_DATASET_ROOT = Path("/root/k8srca/Cloud-OpsBench_history")
DEFAULT_SAVE_ROOT = Path("/root/whyfail")


def normalize_system_name(system: str) -> str:
    system_key = (system or "").strip().lower()
    if system_key in {"trainticket", "train-ticket"}:
        return "trainticket"
    if system_key == "boutique":
        return "boutique"
    raise ValueError(f"Unsupported system: {system}")


def resolve_workspace_path(diag_conf: Dict[str, Any], normalized_system: str) -> Path:
    configured = diag_conf.get("workspace_path")
    if configured:
        return Path(configured)

    dataset_root = Path(diag_conf.get("dataset_root", DEFAULT_DATASET_ROOT))
    return dataset_root / "benchmark" / normalized_system


def resolve_save_path(diag_conf: Dict[str, Any], normalized_system: str) -> Path:
    configured = diag_conf.get("save_path")
    if configured:
        return Path(configured)

    save_root = Path(diag_conf.get("save_root", DEFAULT_SAVE_ROOT))
    return save_root / normalized_system


def load_metadata(meta_path: Path) -> Dict[str, Any]:
    """Load metadata.json for one case."""
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_completed_trace(trace_path: Path) -> bool:
    """Return whether a trace records a completed or budget-exhausted run."""
    if not trace_path.exists():
        return False
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return trace.get("stop_reason") in {"final_answer", "max_steps"}


def resolve_case_names(fault_root: Path, diag_conf: Dict[str, Any]) -> List[str]:
    """
    Resolve case names from config.

    Rules:
    - If config specifies a non-empty case_name, run only that case.
    - Otherwise, enumerate all subdirectories under fault_root.
    """
    case_name = diag_conf.get("case_name", None)

    if case_name is not None and str(case_name).strip():
        return [str(case_name).strip()]

    if not fault_root.exists():
        raise FileNotFoundError(f"Fault root not found: {fault_root}")

    case_names = sorted(
        p.name for p in fault_root.iterdir()
        if p.is_dir()
    )

    if not case_names:
        raise ValueError(f"No case directories found under: {fault_root}")

    return case_names


def write_trace_file(trace_path: Path, task: TaskSpec, result: CaseResult, extra_meta: Dict[str, Any]) -> None:
    """
    Persist one case's trace in the same shape evaluation.py already reads,
    no matter which backend produced it.

    - If the backend already has a full legacy trace (e.g. the single-agent
      runtime, which writes its own file incrementally as it runs), reuse it
      as-is and just fold in the run-level bookkeeping (model name, ground
      truth, etc.).
    - Otherwise, assemble the trace from TaskSpec.trace - every tool call any
      agent made, captured automatically, regardless of framework.
    """
    if result.raw_trace is not None:
        payload = dict(result.raw_trace)
        payload["metadata"] = {**payload.get("metadata", {}), **extra_meta}
    else:
        full_question = (
            f"The Kubernetes environment in namespace `{task.namespace}` is experiencing a fault. "
            f"A high-level symptom has been reported: '{task.symptom}'. "
            f"Diagnose the root cause of this incident."
        )
        payload = {
            "case_id": task.case_id,
            "system_name": extra_meta.get("fault_category"),
            "question": full_question,
            "case_path": extra_meta.get("case_path"),
            "max_steps": task.max_tool_calls,
            "current_step": len(task.trace.records),
            "finished": result.finished,
            "final_answer": result.final_answer.model_dump_json() if result.final_answer else None,
            "stop_reason": result.stop_reason,
            "metadata": extra_meta,
            "steps": task.trace.as_steps(),
        }

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_single_case(
    case_name: str,
    workspace_path: str,
    save_path: str,
    fault_category: str,
    model_name: str,
    max_iterations: int,
    llm_conf: Dict[str, Any],
    backend_name: str,
) -> None:
    """
    Run one single case end-to-end, through whichever backend implements
    AgentBackend.solve_case(TaskSpec) -> CaseResult.
    """
    # -------------------------------------------------------------------
    # 1. Resolve paths
    # -------------------------------------------------------------------
    fault_root = Path(workspace_path) / fault_category
    case_path = fault_root / case_name
    meta_path = case_path / "metadata.json"

    if not case_path.exists():
        raise FileNotFoundError(f"Case path not found: {case_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")

    diag_root = Path(save_path) / model_name / fault_category
    diag_case_path = diag_root / case_name
    diag_case_path.mkdir(parents=True, exist_ok=True)

    trace_dir = str(diag_case_path)
    trace_path = diag_case_path / f"{case_name}.json"

    # Skip only completed traces; an interrupted partial trace should be rerun.
    if is_completed_trace(trace_path):
        print(f"[SKIP] Case already has a completed trace: {case_name}")
        return

    # -------------------------------------------------------------------
    # 2. Load metadata
    # -------------------------------------------------------------------
    metadata_data = load_metadata(meta_path)
    query = metadata_data.get("query", "")
    namespace = metadata_data.get("namespace", "")
    result = metadata_data.get("result", "")

    # -------------------------------------------------------------------
    # 3. Build the tools + the structured TaskSpec handed to the backend
    # -------------------------------------------------------------------
    benchmark_system = "train-ticket" if namespace == "train-ticket" else "boutique"
    tools_list = create_k8s_tools(
        str(case_path),
        system=benchmark_system,
        fault_category=fault_category,
    )

    task = build_task_spec(
        tools_list=tools_list,
        case_id=case_name,
        system=benchmark_system,
        namespace=namespace,
        symptom=query,
        max_tool_calls=max_iterations,
        metadata={
            "result": result,
            "fault_category": fault_category,
            "model_name": model_name,
            "run_started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )

    # -------------------------------------------------------------------
    # 4. Run the case through the selected backend
    # -------------------------------------------------------------------
    backend_cls = BACKENDS[backend_name]
    backend = backend_cls(llm_conf=llm_conf, trace_dir=trace_dir)
    case_result = backend.solve_case(task)

    # -------------------------------------------------------------------
    # 5. Persist trace + print summary
    # -------------------------------------------------------------------
    write_trace_file(
        trace_path=trace_path,
        task=task,
        result=case_result,
        extra_meta={
            "namespace": namespace,
            "query": query,
            "result": result,
            "fault_category": fault_category,
            "model_name": model_name,
            "backend": backend_name,
            "case_path": str(case_path),
        },
    )

    print("=" * 80)
    print("Cloud-OpsBench Multi-Backend Runtime")
    print("=" * 80)
    print(f"Model         : {model_name}")
    print(f"Backend       : {backend_name}")
    print(f"Fault Category: {fault_category}")
    print(f"Case Name     : {case_name}")
    print(f"Case Path     : {case_path}")
    print(f"Finished      : {case_result.finished}")
    print(f"Stop Reason   : {case_result.stop_reason}")
    print(f"Tool Calls    : {len(task.trace.records)}")
    print(f"Trace Path    : {trace_path}")
    print("-" * 80)
    print("Final Answer:")
    if case_result.final_answer:
        print(case_result.final_answer.model_dump_json(indent=2))
    else:
        print("[No final answer produced]")
    print("=" * 80)


def main() -> None:
    # -------------------------------------------------------------------
    # 1. Load config
    # -------------------------------------------------------------------
    config = load_config(str(AGENT_ROOT / "configs/model_configs.yaml"))
    llm_conf = config.model
    diag_conf = config.diagnosis

    model_name = llm_conf["model"]
    normalized_system = normalize_system_name(diag_conf["system"])
    workspace_path = resolve_workspace_path(diag_conf, normalized_system)
    save_path = resolve_save_path(diag_conf, normalized_system)
    fault_category = diag_conf["fault_category"]
    max_iterations = diag_conf["max_iterations"]
    backend_name = str(diag_conf.get("backend", "react_single_agent")).strip() or "react_single_agent"
    if backend_name not in BACKENDS:
        raise ValueError(
            f"Unknown diagnosis.backend '{backend_name}'. Available: {sorted(BACKENDS)}"
        )

    # fault_root = Path(workspace_path) / "benchmark" / fault_category
    fault_root = workspace_path / fault_category
    if not fault_root.exists():
        raise FileNotFoundError(f"Fault root not found: {fault_root}")

    # -------------------------------------------------------------------
    # 2. Resolve case list
    # -------------------------------------------------------------------
    case_names = resolve_case_names(fault_root, diag_conf)
    # case_names = case_names[:MAX_CASES_TO_RUN]

    print("✅ Configuration loading completed")
    print(f"Model          : {model_name}")
    print(f"Backend        : {backend_name}")
    print(f"Fault category : {fault_category}")
    print(f"Workspace path : {workspace_path}")
    print(f"Save path      : {save_path}")
    print(f"Max iterations : {max_iterations}")
    print(f"Max cases      : {MAX_CASES_TO_RUN}")
    print(f"Case count     : {len(case_names)}")
    print(f"Cases          : {case_names}")

    # -------------------------------------------------------------------
    # 3. Run all selected cases
    # -------------------------------------------------------------------
    for idx, case_name in enumerate(case_names, start=1):
        print(f"\n[RUN] ({idx}/{len(case_names)}) case={case_name}")
        try:
            run_single_case(
                case_name=case_name,
                workspace_path=str(workspace_path),
                save_path=str(save_path),
                fault_category=fault_category,
                model_name=model_name,
                max_iterations=max_iterations,
                llm_conf=llm_conf,
                backend_name=backend_name,
            )
        except Exception as e:
            print(f"[ERROR] case={case_name} failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)
