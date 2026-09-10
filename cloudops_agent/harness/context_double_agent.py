from __future__ import annotations

from pathlib import Path

from runtime.core import CaseState, StepRecord


class DoubleAgentContextBuilder:
    """Build the complete text prompt while retaining the full ReAct history."""

    def __init__(self, *, tools_description: str, expected_output_diagnostic: str, expected_output_verifier: str):
        self.tools_description = tools_description
        self.expected_output_diagnostic = expected_output_diagnostic
        self.expected_output_verifier = expected_output_verifier
        self.diagnostic_prompt = Path(__file__).with_name("diagnostic.md").read_text(
            encoding="utf-8"
        ).strip()
        self.verifier_prompt = Path(__file__).with_name("verifier.md").read_text(
            encoding="utf-8"
        ).strip()


    def build_diagnostic(self, state: CaseState, verifier_answer: str) -> str:
        sections = [
            self.diagnostic_prompt,
            "## Available Tools\nYou may use exactly one tool per step.\n"
            + self.tools_description,
            "## Final Diagnosis Output Requirement\n"
            "When you decide to finish, call `Submit`. Its Action Input must "
            "strictly follow this specification.\n"
            + self.expected_output_diagnostic.strip(),
            self._history(state),
            self._build_verifier_agent_section(verifier_answer),
            "## Current Case\n"
            f"Question: {state.question}\n"
            f"Current Step: {state.current_step + 1}\n"
            f"Budget Steps: {state.max_steps}",
        ]
        return "\n\n".join(section for section in sections if section.strip())

    def _build_verifier_agent_section(self, verifier_answer: str) -> str:
        if not verifier_answer:
            return ""

        return "A Verifier Agent analysed your last steps and replied:\n" \
        f"{verifier_answer}\n\n"

    def build_verifier(self, state: CaseState, diagnostic_answer: str) -> str:
        sections = [
            self.verifier_prompt,
            # "## Available Tools\nYou may use exactly one tool per step.\n"
            # + self.tools_description,
            "## Output Requirement\n"
            "Your output must strictly follow this specification.\n"
            + self.expected_output_verifier.strip(),
            self._history(state),
            self._build_proposed_diagnosis_section(diagnostic_answer),
            "## Current Case\n"
            f"Question: {state.question}\n"
            f"Current Step: {state.current_step + 1}\n"
            f"Budget Steps: {state.max_steps}",
        ]
        return "\n\n".join(section for section in sections if section.strip())

    def _build_proposed_diagnosis_section(self, diagnostic_answer: str) -> str:
        if not diagnostic_answer:
            return ""
        return (
            "## Proposed Diagnosis (under review)\n"
            "This is the diagnosis you are reviewing right now - it is NOT yet part of the evidence trail above:\n"
            f"{diagnostic_answer}"
        )
    
    @staticmethod
    def _history(state: CaseState) -> str:
        if not state.history:
            return "## Previous Steps\nNone yet."
        return "## Previous Steps\n\n" + "\n\n".join(
            DoubleAgentContextBuilder._format_step(step) for step in state.history
        )

    @staticmethod
    def _format_step(step: StepRecord) -> str:
        parts = [f"Step {step.step_id}"]
        if step.thought:
            parts.append(f"Thought: {step.thought}")
        if step.action_type in {"tool", "submit"}:
            parts.append(f"Action: {step.action_name}")
            parts.append(f"Action Input: {step.action_input}")
        if step.observation is not None:
            parts.append(f"Observation: {step.observation}")
        if step.error:
            parts.append(f"Error: {step.error}")
        return "\n".join(parts)
