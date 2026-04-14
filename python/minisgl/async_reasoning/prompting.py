# thinker (t) is worker 0, writer (w) is worker 1, mode switching is queried separately
# thinker sees:    "{input_prompt}{thinker_output}"
# writer sees:     "{input_prompt}{thinker_output}{writer_output}
# mode switching:  "{mode_switching_prompt}{thinker_output}{writer_output}


class AsyncReasoningPrompting:
    def __init__(self, problem):
        self.input_prompt = f"""
<|im_start|>user
{problem}
""".strip() + "\n"
        # writer_output and thinker_output starts with these prefixes
        self.thinker_output_prefix = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.writer_output_prefix = f""" ... [SYSTEM: thoughts will continue here]\n</think>\n"""

        self.mode_switching_prompt = f"""
<|im_start|>user
You are an AI assistant that can think and write responses concurrently, and you must decide whether or not you should pause writing and think more.
Read the current partial thoughts and response below, then decide whether you can continue writing the response without pausing (yes/no):
 - Answer "yes" if your thoughts have enough information to write the next response paragraph, even if the full task is not solved yet.
 - Answer "no" if your thoughts aren't enough to write the next response paragraph, i.e. if your response ran out of of thoughts.
""".strip() + "\n"

        # these questions are inserted to change mode depending on model answers
        self.mode_switching_question = "...\n\nWait, are my current thoughts enough to write the next paragraph or formula? (yes/no): "
        self.yes_token, self.no_token = "yes", "no"
