"""Two short text-only Qwen requests with a fixed choice set."""
import json

from .mission_context import MEMORY, repeated_return

MEMORY_PROMPT = (
    "Decide whether this measured navigation event should be retained to help the rest of the task. "
    "Save a consequential block, failed progress or retraced route. Skip routine events and duplicates. "
    "A temporary block is not proof of a permanent dead end. Reply K to keep or S to skip."
)
SUBGOAL_PROMPT = (
    "Choose the local route serving the user's current unfinished task. Each letter is a route already checked "
    "on the measured map. Compare actual travel bearing, prior attempts, camera clues and visit counts. "
    "For exploration, prefer a reachable branch not yet tried. For object tasks preserve the requested target. "
    "Reply S whenever direct action selection is better, including precision approach, a requested rotation or "
    "wait, uncertain goal progress, or a visually indicated hazard. Never claim task completion. Reply with one letter."
)


async def evaluate_mission(policy, packet):
    is_memory = packet["protocol"] == MEMORY
    envelope = {key: packet[key] for key in ("protocol", "snapshot_id", "map_version")}
    if is_memory and packet["event"]["kind"] == "returned":
        choice = "S" if repeated_return(packet["event"], packet["notes"]) else "K"
        return {**envelope, "choice": choice, "source": "measured_return"}
    keys = ("instruction", "event", "notes") if is_memory else ("instruction", "routes", "notes", "view", "progress")
    context = json.dumps({k: packet[k] for k in keys}, ensure_ascii=False, separators=(",", ":"))
    prompt = MEMORY_PROMPT if is_memory else SUBGOAL_PROMPT
    tokens = policy.encode(
        f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{context}"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
    )
    if len(tokens) > 1500:
        raise ValueError("Mission packet exceeds its token budget")
    labels = ["K", "S"] if is_memory else sorted(packet["routes"]) + ["S"]
    block = await policy.llm.create_block()
    try:
        answer = await policy.llm.forward(tokens, cache_view=[block])
        probs = policy.label_distribution(answer.logits, labels, error="Non-finite mission logits").probabilities
        choice = labels[max(range(len(labels)), key=probs.__getitem__)]
    finally:
        await policy.llm.free_block(block)
    return {**envelope, "choice": choice, "source": "qwen"}
