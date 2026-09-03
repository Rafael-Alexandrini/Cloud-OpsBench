#!/usr/bin/env python
"""
Print the exact prompt the single-agent runtime (runtime/prompt_builder.py)
would send to the LLM for step 1 of a given case - no API call, just prompt
construction. Useful to eyeball the real prompt size/content without
spending a request on your remote model.

Usage:
    python print_prompt.py [system] [fault_category] [case_name]

Defaults to boutique/runtime/22 (this repo's example case).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from prompts.RCA_candidate import agent_prompt, build_expected_output
from runtime.prompt_builder import PromptBuilder
from runtime.state import init_case_state
from tools.definition import create_k8s_tools
from tools.registry import build_tool_registry, render_tools_description


def build_prompt_for_case(system: str, fault_category: str, case_name: str) -> str:
    repo_root = AGENT_ROOT.parent
    case_path = repo_root / "benchmark" / system / fault_category / case_name
    meta_path = case_path / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    namespace = metadata.get("namespace", "")
    query = metadata.get("query", "")

    full_question = (
        f"The Kubernetes environment in namespace `{namespace}` is experiencing a fault. "
        f"A high-level symptom has been reported: '{query}'. "
        f"Diagnose the root cause of this incident."
    )

    tools_list = create_k8s_tools(str(case_path), system=system, fault_category=fault_category)
    tool_registry = build_tool_registry(tools_list)
    tools_description = render_tools_description(tool_registry)

    prompt_builder = PromptBuilder(
        tools_description=tools_description,
        backstory_prompt=agent_prompt,
        expected_output=build_expected_output(system),
    )

    # Fresh case at step 1: empty history, same as what run.py builds before
    # ever calling the model.
    state = init_case_state(
        case_id=case_name,
        system_name=fault_category,
        question=full_question,
        case_path=str(case_path),
        max_steps=20,
    )

    return prompt_builder.build(state)


def main() -> None:
    system = sys.argv[1] if len(sys.argv) > 1 else "boutique"
    fault_category = sys.argv[2] if len(sys.argv) > 2 else "runtime"
    case_name = sys.argv[3] if len(sys.argv) > 3 else "22"

    prompt = build_prompt_for_case(system, fault_category, case_name)

    print(f"# system={system} fault_category={fault_category} case={case_name}", file=sys.stderr)
    print(f"# prompt length: {len(prompt)} chars", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(prompt)


if __name__ == "__main__":
    main()
