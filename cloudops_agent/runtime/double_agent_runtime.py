from __future__ import annotations

from .state import CaseState, StepRecord, append_step


class DoubleAgentRuntime:
    def __init__(
        self,
        prompt_builder,
        model_runner,
        output_parser,
        tool_executor,
        trace_logger,
    ):
        """
        Args:
            prompt_builder: PromptBuilder instance
            model_runner: ModelRunner instance
            output_parser: OutputParser instance
            tool_executor: ToolExecutor instance
            trace_logger: TraceLogger instance
        """
        self.prompt_builder = prompt_builder
        self.model_runner = model_runner
        self.output_parser = output_parser
        self.tool_executor = tool_executor
        self.trace_logger = trace_logger


    def run_case(self, state: CaseState) -> CaseState:
        """
        Run a single case until finish or max_steps.

        Args:
            state: Initialized case state.

        Returns:
            Final case state after execution.
        """
        verif_counter = 0
        max_verif_turns = 3
        verif_answer = ""
        state.metadata["verif_model_latency"] = 0
        state.metadata["verif_input_tokens"] = 0
        state.metadata["verif_output_tokens"] = 0
        while not state.finished and state.current_step < state.max_steps:
            step_id = state.current_step + 1
            prompt = self.prompt_builder.build(state, verif_answer) 
            verif_answer = ""
            # 1) Call model
            raw_model_output = ""
            model_latency = None
            input_tokens = None
            output_tokens = None
            
            try:
                model_result = self.model_runner.generate(prompt)
                raw_model_output = model_result.get("text", "")
                model_latency = model_result.get("latency")
                input_tokens = model_result.get("input_tokens")
                output_tokens = model_result.get("output_tokens")
            except Exception as e:
                step_record = StepRecord(
                    step_id=step_id,
                    prompt=prompt,
                    raw_model_output=raw_model_output,
                    thought=None,
                    action_type="invalid",
                    action_name=None,
                    action_input=None,
                    final_answer=None,
                    observation=None,
                    error=f"ModelRunner error: {e}",
                    model_latency=model_latency,
                    tool_latency=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                append_step(state, step_record)
                self.trace_logger.save_case_state(state)
                state.current_step += 1
                continue
                # 2) Parse output
            parsed = self.output_parser.parse(raw_model_output)
        
            thought = parsed.get("thought")
            action_type = parsed.get("type")
            action_name = parsed.get("action_name")
            action_input = parsed.get("action_input")
            final_answer = parsed.get("final_answer")
            parse_error = parsed.get("error")
        
            # 3) Final answer path
            if action_type == "finish":
                # chama agente validador
                # se válido, retorna StepRecord, se inválido, adiciona no prompt do agente principal e volta nesse loop
                # se agente principal responder com finish e validador bloquear 3 vezes, retornar StepRecord
            

                if verif_counter >= max_verif_turns:
                    step_record = StepRecord(
                        step_id=step_id,
                        prompt=prompt,
                        raw_model_output=raw_model_output,
                        thought=thought,
                        action_type="finish",
                        action_name=None,
                        action_input=None,
                        final_answer=final_answer,
                        observation=None,
                        error=None,
                        model_latency=model_latency,
                        tool_latency=None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    append_step(state, step_record)
                    state.finished = True
                    state.final_answer = step_record.final_answer
                    state.stop_reason = "final_answer"
                    self.trace_logger.save_case_state(state)
                    state.current_step += 1
                    continue

                # Chama verificador
                prompt_verif = self.prompt_builder.build_validator_prompt(state, final_answer)
                try:
                    verif_result = self.model_runner.generate(prompt_verif)
                    verif_raw_model_output = verif_result.get("text", "")
                    verif_model_latency = verif_result.get("latency")
                    verif_input_tokens = verif_result.get("input_tokens")
                    verif_output_tokens = verif_result.get("output_tokens")
                except Exception as e: 
                    # Erro ao chamar validador, aceita resposta final.               
                    print("Erro ao chamar validador!\n", e)
                    step_record = StepRecord(
                        step_id=step_id,
                        prompt=prompt,
                        raw_model_output=raw_model_output,
                        thought=thought,
                        action_type="finish",
                        action_name=None,
                        action_input=None,
                        final_answer=final_answer,
                        observation=None,
                        error=None,
                        model_latency=model_latency,
                        tool_latency=None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    append_step(state, step_record)
                    state.finished = True
                    state.final_answer = step_record.final_answer
                    state.stop_reason = "final_answer"
                    self.trace_logger.save_case_state(state)
                    state.current_step += 1
                    continue

                state.metadata["verif_model_latency"] += verif_model_latency
                state.metadata["verif_input_tokens"] += verif_input_tokens
                state.metadata["verif_output_tokens"] += verif_output_tokens
                verif_parsed = self.output_parser.parse(verif_raw_model_output)
                        
                verif_thought = verif_parsed.get("thought")
                verif_action_type = verif_parsed.get("type")
                verif_action_name = verif_parsed.get("action_name")
                verif_action_input = verif_parsed.get("action_input")
                verif_final_answer = verif_parsed.get("final_answer")
                verif_parse_error = verif_parsed.get("error")

                if (verif_action_name or "").strip().strip(".").lower() == "disagree":
                    # adiciona no histórico do agente a proposta
                    verif_answer = (verif_action_input or {}).get("reason") or (
                        "The validator disagreed with your diagnosis but did not provide a specific reason."
                    )
                    verif_counter += 1
                    continue

                else: # Agree ou erro no verificador 
                    step_record = StepRecord(
                        step_id=step_id,
                        prompt=prompt,
                        raw_model_output=raw_model_output,
                        thought=thought,
                        action_type="finish",
                        action_name=None,
                        action_input=None,
                        final_answer=final_answer,
                        observation=None,
                        error=None,
                        model_latency=model_latency,
                        tool_latency=None,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    append_step(state, step_record)
                    state.finished = True
                    state.final_answer = step_record.final_answer
                    state.stop_reason = "final_answer"
                    self.trace_logger.save_case_state(state)
                    state.current_step += 1
                    continue

            # 4) Invalid path
            elif action_type == "invalid":
                step_record = StepRecord(
                    step_id=step_id,
                    prompt=prompt,
                    raw_model_output=raw_model_output,
                    thought=thought,
                    action_type="invalid",
                    action_name=action_name,
                    action_input=action_input,
                    final_answer=None,
                    observation=None,
                    error=parse_error or "Invalid model output.",
                    model_latency=model_latency,
                    tool_latency=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                append_step(state, step_record)
                self.trace_logger.save_case_state(state)
                state.current_step += 1
                continue

            # 5) Tool execution path
            elif action_type == "tool":
                tool_result = self.tool_executor.execute(action_name, action_input)
                observation = tool_result.get("observation")
                tool_error = tool_result.get("error")
                tool_latency = tool_result.get("latency")
        
                step_record = StepRecord(
                    step_id=step_id,
                    prompt=prompt,
                    raw_model_output=raw_model_output,
                    thought=thought,
                    action_type="tool",
                    action_name=action_name,
                    action_input=action_input,
                    final_answer=None,
                    observation=observation,
                    error=tool_error,
                    model_latency=model_latency,
                    tool_latency=tool_latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                append_step(state, step_record)
                self.trace_logger.save_case_state(state)
                state.current_step += 1



        if not state.finished and state.current_step >= state.max_steps:
            state.stop_reason = "max_steps"
            self.trace_logger.save_case_state(state)

        return state

    