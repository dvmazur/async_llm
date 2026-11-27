"""Health Gathering H+Delta roles, ported from the supplied gist.

Source: timka-byzov/3cc4a875991b897eec6131b14869b49e at a948856.
UI side effects are removed; calls use a recording adapter and greedy policy.
Parallel prefix/tail/checkpoint visibility and H/Delta operations are retained.
"""

from __future__ import annotations

import asyncio

import torch


def make_prompting(llm, cfg):

    class prompting:
        actions = {
            "wait": "do nothing while the game continues.",
            "forward": "move forward in the direction you are facing.",
            "right": "turn your point-of-view character right.",
            "left": "turn your point-of-view character left.",
        }
        env_specific_hint = "Your goal is to survive as long as possible. The damaging floor drains health, while health packs restore it. Use screen coordinates from the current (second) frame: a health pack left of center requires `left`, one right of center requires `right`, and a centered health pack requires `forward`. If none is visible, turn to search. The current frame overrides history."
        _action_info = "\n".join(
            (
                f"- `{action_name}`: {action_info.strip()}"
                for (action_name, action_info) in actions.items()
            )
        )
        _action_options = ", ".join((f"`{a}`" for a in actions))
        assistant_turn_start = "<|im_start|>assistant\n<think>\n\n</think>\n"
        turn_end = "\n<|im_end|>\n"
        turn_end_token_id = llm.tokenizer.vocab["<|im_end|>"]
        action_names = list(actions.keys())
        action_token_ids = [llm.tokenizer.vocab[a] for a in actions]
        decision_names = ["continue", *action_names]
        decision_token_ids = [llm.tokenizer.vocab[a] for a in decision_names]
        _decision_options = ", ".join((f"`{a}`" for a in decision_names))
        eos_token_id = llm.tokenizer.vocab["<|im_end|>"]
        common_prompt = llm.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"\nYou will be playing a videogame called Doom from a first-person perspective. The videogame requires that you observe the screen as image inputs and choose one of the following actions: {_action_info}\nGame time advances only when an environment action is executed; `continue` does not advance it. Another instance of you will periodically read your reasoning to choose its next action. This will happen repeatedly as the episode progresses.\nYour thoughts should be compressed paragraphs because of time constraints - think about what you should do in this situation, do not spend time double-checking everything.\n{env_specific_hint}".strip(),
                }
            ],
            add_generation_prompt=False,
        )
        history_init = assistant_turn_start + "Previous observations are listed below:\n"
        describer_init = (
            "First frame is previous; second is current. In one complete sentence of at most four words, report only the most important visible fact about health-pack position relative to screen center or its screen-space change. Left and right always mean screen coordinates. Do not infer, name, or explain the player's action. End with a period, then stop. Output no heading."
            + turn_end
            + assistant_turn_start
        )
        actor_init = (
            "These are the last two images (previous, then current). Use the current frame as the source of truth. Apply exactly: health pack left -> `left`; health pack right -> `right`; centered health pack -> `forward`; none visible -> turn to search. Say what you should do and choose action in one quick sentence."
            + turn_end
            + assistant_turn_start
            + "The next thing I should do is"
        )
        mailbox_bridge = turn_end + "<|im_start|>user\n"
        actor_probe_prompt = f"\nUse the current frame as the source of truth. Apply exactly: health pack left -> `left`;\nhealth pack right -> `right`; centered health pack -> `forward`; none visible -> turn to search.\n...\n**Actor:** I must now choose the action, based on the current game state (image) and the partial reasoning above.\nMy options are: {_decision_options}. `continue` means keep reasoning without taking a game action yet.\n\nAction: `\n".strip()

        @classmethod
        def get_image_dict(
            cls, *images, without_eot: bool = False, end_previous_turn: bool = False
        ) -> dict[str, torch.Tensor]:
            """Convert image into a VLM-friendly dictionary of tensors that ends BEFORE eos"""
            input_dict = llm.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [*[{"type": "image", "image": image} for image in images]],
                    }
                ],
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images_kwargs={"size": {"shortest_edge": 65536, "longest_edge": cfg.max_pixels}},
            )
            eot_id = cls.turn_end_token_id
            if without_eot:
                end_of_turn_pos = input_dict["input_ids"][0].tolist().index(eot_id)
                for field in ("input_ids", "mm_token_type_ids", "attention_mask"):
                    input_dict[field] = input_dict[field][..., :end_of_turn_pos]
            if end_previous_turn:
                input_dict["input_ids"] = torch.cat(
                    [input_dict["input_ids"].new_tensor([[eot_id]]), input_dict["input_ids"]], dim=1
                )
                input_dict["mm_token_type_ids"] = torch.cat(
                    [
                        input_dict["mm_token_type_ids"].new_tensor([[0]]),
                        input_dict["mm_token_type_ids"],
                    ],
                    dim=1,
                )
                input_dict["attention_mask"] = torch.cat(
                    [input_dict["attention_mask"].new_tensor([[1]]), input_dict["attention_mask"]],
                    dim=1,
                )
            return input_dict

    return prompting


async def parallel(*jobs):
    group = asyncio.gather(*jobs, return_exceptions=True)
    try:
        results = await asyncio.shield(group)
    except BaseException:
        await asyncio.shield(group)
        raise
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


async def free_blocks(llm, blocks):
    await parallel(*(llm.free_block(block) for block in blocks))


async def choose_action(llm, prompting, prompt, history, delta, images, cfg):

    def tokenize(text):
        return llm.tokenizer(
            text, return_tensors="pt", add_special_tokens=False, padding="do_not_pad"
        )

    def visible_text(token):
        llm.emit(token)
        text = llm.tokenizer.decode([token.item()])
        return "" if token.item() == prompting.eos_token_id or "<|im_" in text else text

    async def sample_descriptor(output):
        return await llm.sample(output, role="descriptor")

    def descriptor_will_stop(token, text, count):
        return (
            token.item() == prompting.eos_token_id
            or "\n" in text
            or (count > 50 and "." in text)
            or (count >= cfg.description_length)
        )

    def thinker_will_stop(token, text):
        return token.item() == prompting.eos_token_id or "<|im_end|>" in text

    probes = []
    (D, d, U, T, t, R) = await parallel(*(llm.create_block() for _ in range(6)))
    round_blocks = [D, d, U, T, t, R]
    try:
        (descriptor_start, thinker_start) = await parallel(
            llm(
                **tokenize(prompting.describer_init), cache_view=[prompt, history, delta, images, D]
            ),
            llm(**tokenize(prompting.actor_init), cache_view=[prompt, history, delta, images, T]),
        )
        descriptor_token = await llm.sample(descriptor_start, role="descriptor")
        thinker_token = await llm.sample(thinker_start, role="thinker")
        (descriptor_step, thinker_step, _) = await parallel(
            llm(descriptor_token.view(1), cache_view=[prompt, history, delta, images, D, d]),
            llm(thinker_token.view(1), cache_view=[prompt, history, delta, images, T, t]),
            llm(
                **tokenize(prompting.mailbox_bridge),
                cache_view=[prompt, history, delta, images, D, U],
                return_logits=False,
            ),
        )
        description_pieces = [visible_text(descriptor_token)]
        thought_pieces = [visible_text(thinker_token)]
        descriptor_count = thinker_count = 1
        last_thinker_text = llm.tokenizer.decode([thinker_token.item()])
        descriptor_token = await sample_descriptor(descriptor_step)
        thinker_token = await llm.sample(thinker_step, role="thinker")
        next_checkpoint = cfg.readiness_interval
        mailbox_published = False
        descriptor_active = True
        while True:
            descriptor_text = (
                llm.tokenizer.decode([descriptor_token.item()]) if descriptor_active else ""
            )
            thinker_text = llm.tokenizer.decode([thinker_token.item()])
            descriptor_terminal = descriptor_active and descriptor_will_stop(
                descriptor_token, descriptor_text, descriptor_count + 1
            )
            thinker_terminal = thinker_will_stop(thinker_token, thinker_text)
            hard_limit = thinker_count + 1 >= cfg.thought_length
            checkpoint = (
                thinker_count >= next_checkpoint
                or "." in last_thinker_text
                or thinker_terminal
                or hard_limit
            )
            if checkpoint:
                t1 = await llm.create_block()
                d1 = await llm.create_block() if descriptor_active else None
                round_blocks.extend((block for block in (d1, t1) if block is not None))
                descriptor_chain = [D, *([d] if d is not None else [])]
                shared_context = [prompt, history, delta, images, *descriptor_chain, U, T, t]
                write_blocks = [t1, R, *([d1] if d1 is not None else [])]
                write_ids = {id(block) for block in write_blocks}
                assert len(write_ids) == len(write_blocks)
                assert not write_ids.intersection(map(id, shared_context))
                checkpoint_jobs = []
                if descriptor_active:
                    checkpoint_jobs.append(
                        llm(
                            descriptor_token.view(1),
                            cache_view=[prompt, history, delta, images, *descriptor_chain, d1],
                        )
                    )
                checkpoint_jobs.extend(
                    (
                        llm(thinker_token.view(1), cache_view=[*shared_context, t1]),
                        llm(
                            **tokenize(prompting.actor_probe_prompt),
                            cache_view=[*shared_context, R],
                        ),
                    )
                )
                checkpoint_results = await parallel(*checkpoint_jobs)
                if descriptor_active:
                    (descriptor_step, thinker_step, readiness_step) = checkpoint_results
                    description_piece = visible_text(descriptor_token)
                    description_pieces.append(description_piece)
                    descriptor_count += 1
                else:
                    (thinker_step, readiness_step) = checkpoint_results
                thought_piece = visible_text(thinker_token)
                thought_pieces.append(thought_piece)
                thinker_count += 1
                decision_logits = readiness_step.logits[prompting.decision_token_ids].float()
                force_action = thinker_terminal or hard_limit
                if force_action:
                    decision_logits[0] = -torch.inf
                decision_probs = decision_logits.softmax(0).cpu()
                decision_index = decision_probs.argmax().item()
                probabilities = dict(zip(prompting.decision_names, decision_probs.tolist()))
                llm.record_decision(readiness_step, prompting.decision_token_ids[decision_index])
                probes.append({"probabilities": probabilities, "forced_action": force_action})
                if decision_index != 0:
                    return (
                        decision_index - 1,
                        "".join(description_pieces),
                        "".join(thought_pieces),
                        probes,
                        round_blocks,
                    )
                if d is not None:
                    await llm.append_block(D, d)
                await llm.append_block(T, t)
                await parallel(*(llm.free_block(block) for block in (d, t, R) if block is not None))
                (d, t) = (d1, t1)
                if descriptor_active and descriptor_terminal:
                    await llm.append_block(D, d1)
                    await llm.free_block(d1)
                    d = None
                    descriptor_active = False
                elif descriptor_active:
                    descriptor_token = await sample_descriptor(descriptor_step)
                thinker_token = await llm.sample(thinker_step, role="thinker")
                last_thinker_text = thought_piece
                next_checkpoint += cfg.readiness_interval
                mailbox_published = True
                continue
            thinker_view = (
                [prompt, history, delta, images, D, U, T, t]
                if mailbox_published
                else [prompt, history, delta, images, T, t]
            )
            if descriptor_active:
                (descriptor_step, thinker_step) = await parallel(
                    llm(
                        descriptor_token.view(1), cache_view=[prompt, history, delta, images, D, d]
                    ),
                    llm(thinker_token.view(1), cache_view=thinker_view),
                )
                description_piece = visible_text(descriptor_token)
                description_pieces.append(description_piece)
                descriptor_count += 1
            else:
                thinker_step = await llm(thinker_token.view(1), cache_view=thinker_view)
            thought_piece = visible_text(thinker_token)
            thought_pieces.append(thought_piece)
            thinker_count += 1
            last_thinker_text = thinker_text
            if descriptor_active and descriptor_terminal:
                descriptor_active = False
            elif descriptor_active:
                descriptor_token = await sample_descriptor(descriptor_step)
            thinker_token = await llm.sample(thinker_step, role="thinker")
    except BaseException:
        await free_blocks(llm, round_blocks)
        raise


async def commit_history(
    llm, prompting, prompt, history, delta, step, description, thought, action, outcome, cfg
):
    record = f"\nObservation: {description.strip()}\nThought: {thought.strip()}\nAction: {action}\nOutcome: {outcome}\n"
    inputs = llm.tokenizer(
        record, return_tensors="pt", add_special_tokens=False, padding="do_not_pad"
    )
    await llm(**inputs, cache_view=[prompt, history, delta], return_logits=False)
    consolidated = step % cfg.consolidation_steps == 0
    if consolidated:
        await llm.append_block(history, delta)
        await llm.free_block(delta)
    return consolidated
