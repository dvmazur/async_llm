from __future__ import annotations

import time
from typing import Any, Optional

DEFAULT_PROMPTS: list[str] = [
    "Write a short paragraph about the ocean.",
    "Explain what a hash map is.",
    "Describe the water cycle in one paragraph.",
]

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

# Surfaced verbatim into the round prompt via agent.py's generic
# task_env.doc hook (see SelfEvolvingAgent.run_step) -- this is the only
# place the "first non-thinking token" scoring rule is explained; there is
# no harness-side enforcement of how act() must be structured, so the agent
# is free to skip <think> entirely, or use it and budget it however it likes.
DOC = (
    "`ttft` scores -(time to your first non-thinking token), not raw time-to-first-token: if the "
    "text your `act()` streams opens with a `<think>...</think>` block, only tokens emitted after "
    "the closing `</think>` count toward this timer -- reasoning is free to take as long as you let "
    "it, but every second spent inside `<think>` counts against you. If you never open a `<think>` "
    "block at all, the very first token you stream counts immediately. This means there is a real "
    "trade-off for you to design, not a fixed rule to satisfy: more reasoning before answering can "
    "raise `math`-style accuracy but costs you here, and skipping it entirely helps this score but "
    "may cost you elsewhere. Decide the balance yourself -- e.g. via how much of your own thinking "
    "you route through a literal `<think>` block versus straight into the answer.\n"
    "This is a speed score for an actual answer, not a reward for responding fast: an episode whose "
    "streamed answer (everything after `</think>`, or the whole stream if you never open one) is "
    "empty, too short to be a real answer, or just the input prompt echoed back unchanged, is scored "
    "the same as never closing `<think>` -- infinitely bad. Being instant is only worth anything if "
    "you actually answered the question."
)


class TTFTEnv:
    """Measures time to the first non-thinking token: on_token(token) is fed
    every token act() streams, in order, and watches for a <think>...</think>
    span at the start of the stream. Tokens inside that span don't stop the
    clock -- only the first token seen once (or if) </think> has closed does.
    act() is never required to use <think> at all -- an implementation that
    streams straight to its answer is timed from its very first token, same
    as before. See DOC above for how this is explained to the agent itself;
    there is no other enforcement of reasoning structure here."""

    name = "ttft"
    doc = DOC

    # Seen live: an act() that skips generation entirely and just echoes the
    # prompt straight back (e.g. `on_token(str(observation)); return
    # str(observation)`) drives ttft to ~microseconds -- technically fast,
    # but it never answered anything. An answer shorter than this many
    # (stripped) characters isn't a real response either way, so there's no
    # need to try to enumerate every way of gaming this -- anything this
    # trivial is caught regardless of *how* it was produced.
    MIN_ANSWER_CHARS = 8

    def __init__(self, prompts: Optional[list[str]] = None) -> None:
        self.prompts = prompts if prompts is not None else DEFAULT_PROMPTS
        self.max_episodes = len(self.prompts)
        self._idx = -1
        self._t_start: Optional[float] = None
        self._t_first: Optional[float] = None
        self._buf = ""
        self._in_think = False
        self._full_buf = ""

    def reset(self) -> Any:
        self._idx = (self._idx + 1) % len(self.prompts)
        self._t_first = None
        self._t_start = time.monotonic()
        self._buf = ""
        self._in_think = False
        self._full_buf = ""
        return self.prompts[self._idx]

    def on_token(self, token: str) -> None:
        # Recorded unconditionally (unlike _buf below, which stops mattering
        # once _t_first is set) so step() can see the actual answer content,
        # not just whether/when it started.
        self._full_buf += token
        if self._t_first is not None:
            return
        self._buf += token
        if not self._in_think and _THINK_OPEN in self._buf:
            self._in_think = True
        if self._in_think:
            if _THINK_CLOSE in self._buf:
                self._in_think = False
                # Approximation: we don't have sub-token timing between
                # </think> and the token after it, so the moment </think>
                # is first seen in the buffer is taken as t_first. If the
                # model never closes the block, t_first stays None and
                # step() below scores it as infinitely bad -- never
                # finishing your reasoning is a real cost, not a loophole.
                self._t_first = time.monotonic()
            return
        self._t_first = time.monotonic()

    def restart(self) -> None:
        """Reset the episode cursor back to the beginning -- see
        MathEnv.restart(). _t_start/_t_first get overwritten by the next
        reset() regardless, but are cleared here too for a clean state."""
        self._idx = -1
        self._t_start = None
        self._t_first = None
        self._buf = ""
        self._in_think = False
        self._full_buf = ""

    def _answered(self) -> bool:
        """True only if the stream contains a real, distinct answer -- not
        empty/trivial, and not just the input prompt echoed back."""
        answer = self._full_buf.split(_THINK_CLOSE, 1)[-1].strip()
        if len(answer) < self.MIN_ANSWER_CHARS:
            return False
        prompt_norm = " ".join(self.prompts[self._idx].split()).strip().lower()
        answer_norm = " ".join(answer.split()).lower()
        return answer_norm != prompt_norm

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        gamed = self._t_first is not None and not self._answered()
        ttft = (self._t_first - self._t_start) if (self._t_first is not None and not gamed) else float("inf")
        return None, -ttft, True, {"ttft": ttft, "gamed": gamed}
