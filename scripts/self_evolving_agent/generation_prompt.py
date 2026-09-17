"""Chat framing for seed and recovery generation of engine revisions."""
def revision_prompt(llm, text):
    return llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)


def revision_completion(prompt, completion):
    # The opening tag belongs to the assistant prefix, not generated tokens.
    # Preserve it so tool-like examples inside thinking cannot execute.
    if prompt.rfind("<think>") > prompt.rfind("</think>"):
        return "<think>" + completion
    return completion
