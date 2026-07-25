"""Prompt strings for the basic Doom VLM demo (reason, then probe for the action).

Two things are asked of the model per env step, from the same prompt:

    reasoning : system + user + <frame> + "what has to happen?"  -> a short,
                abstract, frame-grounded trace.  It is told NOT to name a key:
                the trace is about the *situation*, not the keystroke.
    probe     : the trace, plus PROBE_QUERY, prefilled on top of it.  The action
                is read off the next-token logits over the action words only
                (see ``doom_basic.demo.action_probe``) — the model never has to
                emit a parseable command.

The mechanics paragraph and the two seed phrases are from ``scripts/doom``, where
a sweep settled them.  On its own the model assumes it can walk toward the
monster and then never fires; and this exact "I press:" phrasing commits to
firing once centred, whereas a "Step 1 / Step 2" framing made it oscillate.
"""

from __future__ import annotations

# System turn for the reasoning pass: correct the mechanics the model gets wrong
# on its own, ask for abstract reasoning, and cap the length hard.  Small models
# will happily narrate a whole strategy guide otherwise, and the trace is only
# there to condition the probe.
REASON_SYS_TEXT = (
    "<|im_start|>system\n"
    "You control the player in Doom. A crosshair is fixed at the CENTRE of the screen. "
    "You CANNOT move forward or back; you can only strafe LEFT/RIGHT, or FIRE. "
    "The monster is already in range. To hit it, strafe toward the side it is on until "
    "it sits under the centre crosshair, then fire.\n"
    "Reason about the frame in the abstract: where the monster sits relative to the "
    "crosshair, and what has to happen next. Do NOT name a key — the keystroke is "
    "decided separately. BE BRIEF: one or two short sentences, 30 words at most."
    "<|im_end|>\n"
)

# User turn, split around the frame: <prefix> <image block> <suffix>.
USER_PREFIX = "<|im_start|>user\n"

# ``{history}`` is HISTORY_TEXT or empty.
USER_SUFFIX = (
    "\nThis is the current frame of the game.{history}\n"
    "Briefly: where is the monster relative to the centre crosshair, and what has to "
    "happen next?"
    "<|im_end|>\n<|im_start|>assistant\n"
)

# The reasoning is seeded with this so the trace opens already grounded in the
# frame instead of restating the rules back at us.
REASON_SEED = "Relative to the centre crosshair, the monster is"

HISTORY_TEXT = " Your most recent actions, oldest first: {actions}."

# The probe: appended to the trace, its next-token logits are the action.
PROBE_QUERY = "\nTo put it under the crosshair and shoot, I press:"
