"""Optional record of the readiness votes actually scored by one demo run."""

import json
from pathlib import Path


class ProbeLog:
    def __init__(self, output: Path) -> None:
        if output.exists():
            raise FileExistsError(output)
        self.output = output
        self.events: list[dict] = []
        self.admitted = False
        self.paused_at_paragraph = False
        self.read_lengths: tuple[int, int, int] | None = None

    def observe_view(self, thinker_length: int, writer_length: int, scored_rows: int) -> None:
        self.read_lengths = thinker_length, writer_length, scored_rows

    def record(
        self,
        *,
        source: str,
        yes: float,
        no: float,
        vote: bool,
        before: bool,
        after: bool,
        discarded: str | None,
        thinker_start: int,
        writer_start: int,
        writer_ids: list[int],
    ) -> None:
        if self.read_lengths is None:
            raise RuntimeError("A scored probe is missing its observed block lengths")
        thinker_length, writer_length, scored_rows = self.read_lengths
        committed_writer = max(0, writer_length - writer_start)
        self.events.append(
            {
                "phase": "continuation" if self.admitted else "admission",
                "source": source,
                "yes_logit": yes,
                "no_logit": no,
                "vote_write": vote,
                "writer_allowed_before": before,
                "writer_allowed_after": after,
                "paused_at_writer_paragraph_before": self.paused_at_paragraph,
                "applied": discarded is None,
                "discarded_reason": discarded,
                "thinker_block_tokens": thinker_length,
                "writer_block_tokens": writer_length,
                "scored_input_tokens": scored_rows,
                "committed_thought_tokens": max(0, thinker_length - thinker_start),
                "committed_answer_tokens": committed_writer,
                "sampled_answer_tokens": len(writer_ids),
                "pending_answer_token_id": writer_ids[-1]
                if len(writer_ids) > committed_writer
                else None,
            }
        )
        self.read_lengths = None
        if vote and discarded is None:
            self.admitted = True
            self.paused_at_paragraph = False

    def save(self, *, model: str, problem: str, prompting: str, protocol: str | None) -> None:
        data = {
            "format_version": 1,
            "model": str(model),
            "problem": problem,
            "prompting": prompting,
            "llama_protocol": protocol,
            "events": self.events,
        }
        text = json.dumps(data, ensure_ascii=False, allow_nan=False)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("x", encoding="utf-8") as f:
            f.write(text + "\n")
