"""Non-retryable input assembly failures, shared by model and transport."""


class ContextBudgetExceeded(ValueError):
    code = 'context_budget_exceeded'
    retryable = False

    def __init__(self, input_tokens, input_limit, context_window):
        self.input_tokens = int(input_tokens)
        self.input_limit = int(input_limit)
        self.context_window = int(context_window)
        super().__init__(f'Observation exceeds token budget: {input_tokens} > '
                         f'{input_limit} input tokens (context window {context_window})')


def error_payload(error):
    payload = dict(error=f'{type(error).__name__}: {error}', goal_id='W')
    if isinstance(error, ContextBudgetExceeded):
        payload.update(error_code=error.code, retryable=False,
                       input_tokens=error.input_tokens, input_limit=error.input_limit,
                       context_window=error.context_window)
    if hasattr(error, "compaction_attempts"):
        payload["compaction_attempts"] = error.compaction_attempts
    return payload
