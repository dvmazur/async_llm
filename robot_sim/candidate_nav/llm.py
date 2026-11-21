"""Qwen vision policy with genuinely concurrent, shared-cache AsyncLLM streams.

Two TP ranks execute the same deterministic coroutine schedule. Only rank zero
owns the socket. Observation, spatial reasoning, and the selector share KV/GDN
blocks; the selector reads both analysis streams while they are still running.
No independent Transformers calls or scripted goal selection are substituted.
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import time

import torch
import torch.distributed as dist
from PIL import Image

from . import action_plan, task_policy, target_vision
from .context_budget import ContextBudgetExceeded
from .target_vision_policy import inspect_frame, task_visual_scope
from .target_geometry import estimate, TrajectoryPreference
from .system_prefix import SystemPrefixCache, prefix_ids, split_inputs
from .prompt_state import state_text, budgeted_inputs
from .long_term_memory import MemoryCoordinator
from .sensor_grounding import sensor_context
from .committed_plan import admissible, first_action_labels, orientation, make_plan
from .mission_context import choose_passage, passage_labels, route_labels
from .trajectory_policy import action_labels, translation_options, immediate_mobility
from .prepared_perception import observer_text
from .inference_cache import ExactEncodingCache, decode_image, cache_statistics
from .inference_decisions import checked_argmax, checked_probabilities
from .action_tokens import ActionTokenCodec
from .inference_tuning import graph_options, append_exact_tokens

SYSTEM = """You are the navigation brain of a small rover exploring an initially unknown environment.
You receive two images: IMAGE 1 is the actual front RGB camera; IMAGE 2 is an online lidar/SLAM map.
On the map white is measured free space, black is occupied, gray is unknown. The cyan arrow is the rover.
Lettered circles are reachable candidate viewpoints. Yellow marks the currently executed waypoint.
No hidden world map or true target coordinates are provided. Use measured geometry for distance.
Follow the task. Find objects by actual camera evidence, not by imagining objects in unexplored space.
Completion has priority over moving to a better viewpoint. Once the requested object is visibly
confirmed nearby at the instructed range, stop; do not keep circling around it.
Frontier candidates reveal unknown space. Inspect candidates approach generic RGB/lidar object proposals;
a proposal's color/description is not a confirmed recognition of the requested object.
Avoid repeatedly visiting the same places. On failure choose a different candidate. Plan beyond the next turn.
Other concurrent assistant streams provide camera observations and spatial reasoning in shared memory.
"""


DIRECT_SYSTEM = """You directly drive a small rover searching an initially unknown environment.
IMAGE 1 is actual front RGB; IMAGE 2 is the online measured lidar/SLAM map.
White is free, black occupied, gray unknown; cyan arrow is rover position and heading.
The path shows previous travel; magenta marks generic RGB/lidar object observations.
There are NO waypoints, candidate positions, automatic path planner or steering fallback.
Choose F=forward, L=turn left in place, R=turn right in place, S=stop.
Read action_spec for exact speeds and limits. Turns are relative to the CURRENT heading.
Each action ends before another observation is taken. The rover stops while you think.
Use camera, measured ranges, heading and online map to navigate passages and explore
unseen areas. Plan several turns ahead, escape dead ends, avoid loops and blocked moves.
Never infer target visibility from the instruction. A colored proposal can be a distractor.
Complete the task only when actual camera evidence confirms the requested object nearby.
The measured proximity rule has priority over further exploration. Concurrent camera and
spatial assistant streams share your live memory. No true target coordinates are available.
"""


def one_token(tokenizer, text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError(f"Label {text!r} must be a single token")
    return tokens[0]


def coordinate_digits(prefix):
    """Next decimal digits for a zero-padded three-digit coordinate 000..511."""
    if len(prefix) >= 3 or (prefix and not prefix.isdigit()):
        raise ValueError("Invalid coordinate prefix")
    place = 10 ** (2 - len(prefix))
    base = int(prefix or "0") * place * 10
    return [str(digit) for digit in range(10) if base + digit * place <= 511]


def binary_probabilities(logits, tokenizer):
    """Score Yes/No without losing the model's preferred capitalization."""
    groups = []
    for variants in (("yes", "Yes", "YES"), ("no", "No", "NO")):
        ids = []
        for word in variants:
            encoded = tokenizer.encode(word, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
        if not ids:
            raise ValueError("Tokenizer has no single-token binary labels")
        groups.append(ids)
    scores = (
        torch.stack([torch.logsumexp(logits[ids], dim=0) for ids in groups])
        .softmax(0)
        .cpu()
        .tolist()
    )
    best = [ids[int(logits[ids].argmax())] for ids in groups]
    return scores, best


class AsyncCandidatePolicy:
    def __init__(self, args):
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import AsyncLLM

        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        # Engine owns an explicit TCP rendezvous. torchrun's agent-store flag
        # otherwise turns both ranks into clients of that unstarted TCP store.
        if args.distributed_addr != "env://":
            os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
        self.context_window = getattr(args, "context_window", 8192)
        # Keep simulation defaults; live 12K needs space for the observation,
        # analysis branches and idle cached system variants in the page pool.
        from .hardware_profile import PROFILES
        self.hardware_profile = getattr(args, 'hardware_profile', 'legacy')
        if self.hardware_profile == 'h100':
            os.environ.setdefault('MINISGL_FP8_TILING', 'h100_exact')
            if getattr(args,'trajectory_policy','legacy')=='evidence':
                os.environ.setdefault('MINISGL_VISION_CACHE_MIB','64')
        profile = PROFILES[self.hardware_profile]
        graphs = graph_options(self.hardware_profile, getattr(args,'decode_cuda_graphs',None))
        live_acceleration=(self.hardware_profile=='h100' and self.context_window==12288
                           and getattr(args,'trajectory_policy','legacy')=='evidence')
        prefill_rows=getattr(args,'prefill_rows',None) or (4096 if live_acceleration else profile['prefill_rows'])
        prefill_graphs=getattr(args,'prefill_cuda_graphs',None)
        if prefill_graphs is None:prefill_graphs=live_acceleration
        prefill_graphs=bool(prefill_graphs and graphs['cuda_graph_max_bs'])
        prefill_graph_rows=[64,256,512] if prefill_graphs else None
        self.cache_tokens = 8192 if self.context_window == 8192 else profile['cache_tokens']
        self.gpu_memory_limit_mib = int(os.environ.get('ROBOT_LLM_MEMORY_LIMIT_MIB','0'))
        self.llm = AsyncLLM(
            args.model,
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(self.rank, self.world),
            quantization=None if args.quantization == "none" else args.quantization,
            max_running_req=6,
            num_page_override=self.cache_tokens,
            max_seq_len_override=self.context_window,
            **graphs,
            max_prefill_rows=prefill_rows,
            shared_cuda_graph_max_depth=8,
            shared_cuda_graph_prefill_rows=prefill_graph_rows,
            distributed_addr=args.distributed_addr,
            distributed_timeout=600.0,
            memory_ratio=0.85,
            allocator_memory_limit_mib=self.gpu_memory_limit_mib,
        )
        self.args = args
        self.tokens = self.llm.tokenizer
        self.system_prefix_cache = (None if args.no_system_prefix_cache else SystemPrefixCache(self.llm))
        self.metadata = {
            "model": args.model,
            "quantization": args.quantization,
            "model_name": os.path.basename(args.model),
            "backend": "native_async_llm",
            "tp_size": self.world,
            "hardware_profile": self.hardware_profile,
            "gpu_memory_limit_mib": self.gpu_memory_limit_mib,
            "gpu_physical_ids": os.environ.get('ROBOT_LLM_GPU',''),
            "fp8_backend": os.environ.get('MINISGL_FP8_BACKEND','triton'),
            "fp8_tiling": os.environ.get('MINISGL_FP8_TILING','reference'),
            "fp8_backend_requested": os.environ.get('ROBOT_FP8_BACKEND_REQUEST','triton'),
            "max_prefill_rows": prefill_rows,
            "prefill_cuda_graph_rows":prefill_graph_rows or [],
            "decode_cuda_graph_batches": graphs['cuda_graph_bs'] or [],
            "merged_prepared_observer": getattr(args,'merge_prepared_observer',False),
            "forward_sampling": "legacy_unused_sampling" if self.llm.sample_forward_token else "raw_logits_only",
            "exact_cpu_caches": True,
            "vision_embedding_cache_mib":int(os.environ.get("MINISGL_VISION_CACHE_MIB","0")),
            "early_plan_protocol":"plan_stream_v1",
            "trajectory_token_decoding": "bare_and_space_prefixed_labels_v1",
            "shared_cache": True,
            "strategy4_planning": "committed_spatial_plan_v1",
            "strategy4_streams": ["camera_observer","spatial_assessment","plan_selector","plan_rationale"],
            "trajectory_policy": getattr(args, "trajectory_policy", "legacy"),
            "max_context_tokens": self.context_window,
            "live_input_token_limit": self.context_window - 2048,
            "generation_reserve_tokens": 2048,
            "kv_cache_token_capacity": self.cache_tokens,
            "system_prefix_cache": "live_exact_tokens_v1" if self.system_prefix_cache else "disabled",
            "streams": ["camera_observer", "spatial_reasoner", "goal_selector"],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(self.rank),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "policy_source_sha256": task_policy.source_hash(),
            "fp8_execution": "native"
            if torch.cuda.get_device_capability(self.rank) >= (8, 9)
            else "bf16_tile_emulation",
        }

    def label_distribution(self,logits,labels,*,error):
        codec=getattr(self,'_action_codec',None)
        if codec is None or codec.tokenizer is not self.tokens:
            codec=self._action_codec=ActionTokenCodec(self.tokens)
        return codec.score(logits,labels,error=error)

    def input_token_limit(self, live):
        # The simulator retains its original 6000-token observation budget.
        return getattr(self.args, "context_window", 8192) - 2048 if live else 6000

    def encode(self, text):
        cache = getattr(self, '_encoding_cache', None)
        if cache is None:
            self._encoding_cache = cache = ExactEncodingCache()
        return cache.encode(self.tokens, text)

    async def prefill_observation(self, inputs, system, common, live):
        cache = getattr(self, "system_prefix_cache", None) if live else None
        full_tokens = int(inputs["input_ids"].numel())
        lease = None
        metrics = dict(system_prefix_tokens=0, system_prefix_cache_hit=False,
                       prefilled_system_tokens=0, observation_prefill_tokens=full_tokens,
                       system_prefix_prefill_s=0.0, system_prefix_cache_enabled=cache is not None)
        view = [common]
        tail = dict(inputs)
        if cache is not None:
            ids = prefix_ids(self.llm.processor, system)
            prefix, tail = split_inputs(inputs, ids)
            lease = await cache.acquire(ids, prefix)
            view = [lease.block, common]
            metrics.update(system_prefix_tokens=len(ids), system_prefix_cache_hit=lease.hit,
                           prefilled_system_tokens=0 if lease.hit else len(ids),
                           observation_prefill_tokens=int(tail["input_ids"].numel()),
                           system_prefix_prefill_s=round(lease.prefill_s, 6))
        started = time.monotonic()
        try:
            await self.llm.forward(**tail, cache_view=view[:-1] or None,
                                   write_to=common, return_logits=False)
        except BaseException:
            if lease is not None:
                await cache.release(lease)
            raise
        metrics["observation_prefill_s"] = round(time.monotonic() - started, 6)
        return view, lease, metrics

    async def close(self):
        try:
            if self.system_prefix_cache is not None:
                await self.system_prefix_cache.close()
        finally:
            await self.llm.close()

    async def propose(self, observation):
        """Generate original image coordinates, with an integer output grammar.

        The grammar constrains syntax only: no frontier/anchor positions or
        geometric scores are supplied by the existing CandidateGenerator.
        """
        started = time.monotonic()
        llm = self.llm
        camera, map_image = (decode_image(observation[k]) for k in ("camera", "map_image"))
        state = {k: v for k, v in observation.items() if k not in {"camera", "map_image", "accumulated_map_image", "memory_context", "memory_update", "_memory_status", "control_reference", "response_mode", "_transport_v", "transport_timing"}}
        live_generation = task_policy.is_task_mode(observation)
        if live_generation:state["sensor_grounding"]=sensor_context(observation,camera.size)
        map_images = []
        if live_generation and observation.get('accumulated_map_image'):
            map_images = [dict(type='text',text='IMAGE 3: ACCUMULATED SLAM. Proposal coordinates still refer to IMAGE 2.'),
                          dict(type='image',image=decode_image(observation['accumulated_map_image']))]
            state['image_roles'] = dict(camera=1,local_map=2,accumulated_map=3)
        analysis_budget = max(64, self.args.analysis_tokens) if live_generation else self.args.analysis_tokens
        analysis_ready = min(31 if live_generation else 7, analysis_budget - 1)
        system = (
            "You generate navigation viewpoints for a rover searching an unknown environment. "
            "IMAGE 1 is front RGB; IMAGE 2 is a 512x512 measured map with pixel grid labels. "
            "White is reachable free space for the robot centre, black occupied, gray unknown or unreachable. "
            "The cyan arrow is the rover, blue line its past trajectory, magenta marks measured object hypotheses. "
            "No precomputed candidate points are given. Propose useful new viewpoints yourself. "
            "Choose white free pixels near unexplored boundaries, approach a plausible target, or turn to inspect. "
            "Do not put the robot inside an object or wall. Keep viewpoints diverse, avoid revisits/distractors. "
            "A proposed object is not necessarily the requested target; use actual RGB evidence. "
            "Use the supplied pixel coordinates of the rover and objects as spatial references."
        )
        if task_policy.is_task_mode(observation):
            system = task_policy.GENERATION_SYSTEM
        inputs = llm.processor.apply_chat_template(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": camera},
                        {"type": "image", "image": map_image},
                        *map_images,
                        {"type": "text", "text": state_text(state, live_generation)},
                    ],
                },
            ],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        inputs.pop("attention_mask", None)
        token_limit = self.input_token_limit(live_generation)
        if inputs["input_ids"].numel() > token_limit:
            raise ValueError(f"Generation observation exceeds token budget: {inputs['input_ids'].numel()} > {token_limit} input tokens")
        common, vision, reason, proposals = [await llm.create_block() for _ in range(4)]
        common_view, prefix_lease, prefill_metrics = [common], None, {}
        events, traces, tasks, timeline = [asyncio.Event(), asyncio.Event()], [[], []], [], []

        def prompt(text):
            return self.encode(
                "<|im_start|>user\n"
                + text
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
            )

        def record(event, **extra):
            timeline.append(
                {"event": event, "seconds": round(time.monotonic() - started, 4), **extra}
            )

        async def analyse(index, block, text):
            try:
                output = await llm.forward(prompt(text), cache_view=common_view + [block])
                for step in range(analysis_budget):
                    token = checked_argmax(output.logits, error="Non-finite proposal analysis logits")
                    traces[index].append(token)
                    if step >= analysis_ready:
                        events[index].set()
                    if token == self.tokens.eos_token_id:
                        break
                    output = await llm.forward([token], cache_view=common_view + [block])
            finally:
                events[index].set()
                record("stream_finished", stream=index, tokens=len(traces[index]))

        raw = []
        try:
            common_view, prefix_lease, prefill_metrics = await self.prefill_observation(
                inputs, system, common, live_generation)
            record("observation_prefilled", tokens=int(inputs["input_ids"].numel()), **prefill_metrics)
            tasks = [
                asyncio.create_task(
                    analyse(
                        0,
                        vision,
                        (task_policy.OBSERVER_PROMPT if task_policy.is_task_mode(observation)
                         else "Describe visible target evidence and occlusions in IMAGE 1. Be concise."),
                    )
                ),
                asyncio.create_task(
                    analyse(
                        1,
                        reason,
                        (task_policy.GENERATION_SPATIAL_PROMPT if live_generation
                         else "Suggest useful directions to explore using IMAGE 2, task and history. Think of white reachable viewpoints near unexplored boundaries. Be concise."),
                    )
                ),
            ]
            await asyncio.gather(*(event.wait() for event in events))
            record("proposer_started", analysis_still_running=any(not t.done() for t in tasks))
            prefix = prompt(task_policy.GENERATION_FORMAT_PROMPT if live_generation else (
                "Generate up to eight diverse next viewpoints. Each row is u,v,h,k. "
                "u and v are three-digit zero-padded image pixel coordinates 000..511 (x right, y down). "
                "Example syntax: 256,128,6,F. "
                "h is an image heading 0..7: 0 right,2 down,4 left,6 up. "
                "k is F for exploration, I for inspection, L for a look/rotation. "
                "For L use the rover pixel and a different heading. "
                "Propose positions in WHITE reachable space, not magenta objects or gray space. "
                "Reply with W instead of a row when no more useful viewpoints remain. "
                "Use only rows, no prose. First row:\n"
            ))
            kinds = {"F": "frontier", "I": "inspect", "L": "look"}
            # Qwen tokenizes multi-digit numbers into digits. Restrict syntax
            # one decimal digit at a time, leaving all image locations available.
            last = None
            for index in range(8):
                fields = []
                for field in range(4):
                    value = ""
                    for digit_index in range(3 if field < 2 else 1):
                        allowed = (
                            coordinate_digits(value)
                            if field < 2
                            else [str(v) for v in range(8)]
                            if field == 2
                            else list(kinds)
                        )
                        if field == 0 and digit_index == 0:
                            allowed = allowed + ["W"]
                        ids = [one_token(self.tokens, item) for item in allowed]
                        separator = "" if digit_index else "\n" if field == 0 else ","
                        output = await llm.forward(
                            prefix if last is None else [last] + self.encode(separator),
                            cache_view=common_view + [vision, reason, proposals],
                        )
                        prefix = None
                        best = checked_argmax(output.logits, ids, error="Non-finite coordinate logits")
                        token, last = allowed[best], ids[best]
                        value += token
                        if token == "W":
                            break
                    if value == "W":
                        break
                    fields.append(value)
                if len(fields) != 4:
                    break
                raw.append(
                    {
                        "u": int(fields[0]),
                        "v": int(fields[1]),
                        "heading": int(fields[2]),
                        "kind": kinds[fields[3]],
                    }
                )
            record("proposer_finished", proposals=len(raw))
            await asyncio.gather(*tasks)
            response = {
                "proposals": raw,
                "snapshot_id": observation["snapshot_id"],
                "map_version": observation["map_version"],
                "episode_id": observation["episode_id"],
                "latency_s": time.monotonic() - started,
                "timeline": timeline,
                "input_tokens": int(inputs["input_ids"].numel()),
                "prefill": prefill_metrics,
                "observer": self.tokens.decode(traces[0], skip_special_tokens=True),
                "reasoner": self.tokens.decode(traces[1], skip_special_tokens=True),
                "image_sha256": {
                    "camera": hashlib.sha256(camera.tobytes()).hexdigest(),
                    "map": hashlib.sha256(map_image.tobytes()).hexdigest(),
                    **({'accumulated_map': hashlib.sha256(map_images[-1]['image'].tobytes()).hexdigest()} if map_images else {}),
                },
                "metadata": self.metadata,
            }
            if self.rank == 0:
                print("GENERATION_AUDIT " + json.dumps(response), flush=True)
            return response
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for block in (common, vision, reason, proposals):
                await llm.free_block(block)
            if prefix_lease is not None:
                await self.system_prefix_cache.release(prefix_lease)

    async def decide(self, observation, on_selection=None):
        started = time.monotonic()
        llm = self.llm
        engine_stats = getattr(getattr(llm,'async_engine',None),'statistics',{})
        metrics_before = {**engine_stats,**cache_statistics(self)}
        trajectory = observation.get("protocol") == action_plan.PROTOCOL
        evidence = trajectory and task_policy.is_task_mode(observation) and getattr(self.args, "trajectory_policy", "legacy") == "evidence"
        rationale_enabled = evidence and observation.get("no_plan_rationale") is not True
        direct = trajectory or observation.get("protocol") == "direct_navigation_v1"
        if observation.get("protocol") not in {"candidate_navigation_v1", "direct_navigation_v1", action_plan.PROTOCOL}:
            raise ValueError("Unsupported protocol")
        horizon = action_plan.trajectory_steps(observation["trajectory_steps"]) if trajectory else 1
        candidates = observation.get("candidates", [])
        if direct and candidates:
            raise ValueError("Direct navigation must not receive waypoint candidates")
        labels = ([k for k,v in action_plan.LABELS.items() if v in observation.get("action_spec",{})] if evidence else ["F", "L", "R", "S"]) if direct else [item["id"] for item in candidates]
        turn_labels = [label for label,name in action_plan.LABELS.items()
                       if 'angle_rad' in observation.get('action_spec',{}).get(name,{})]
        if not direct and (
            len(labels) != len(set(labels)) or any(label not in "ABCDEFGH" for label in labels)
        ):
            raise ValueError("Invalid candidate labels")
        camera = decode_image(observation["camera"])
        map_image = decode_image(observation["map_image"])
        split_vision = evidence and target_vision.tracking_enabled(observation)
        visual_state = None
        target_images = []
        if split_vision:
            visual_state = target_vision.initial_state(
                observation['instruction'], observation.get('temporal_visual_state'))
            target_images = target_vision.reference_content(observation, visual_state, decode_image)
        prepared_observer = observer_text(observation, camera) if evidence else None
        previous_camera = observation.get('previous_camera') if evidence and not split_vision and prepared_observer is None else None
        state = {k: v for k, v in observation.items() if k not in {"camera", "map_image", "accumulated_map_image", "previous_camera", "target_reference", "memory_context", "memory_update", "_memory_status", "control_reference", "response_mode", "passage_candidates", "no_plan_rationale", "_transport_v", "transport_timing"}}
        map_images = []
        if task_policy.is_task_mode(observation):
            state['image_roles'] = dict(camera=1,local_map=2)
        if task_policy.is_task_mode(observation) and observation.get('accumulated_map_image'):
            map_images = [dict(type='text',text='IMAGE 3: ACCUMULATED SLAM (historical layout, distinct scale; see maps_context).'),
                          dict(type='image',image=decode_image(observation['accumulated_map_image']))]
            state['image_roles']['accumulated_map'] = 3
        previous_images = []
        if previous_camera is not None:
            state['previous_camera'] = {k:v for k,v in previous_camera.items() if k!='image'}
            old_pose=previous_camera['pose'];new_pose=observation.get('camera_pose',observation.get('perception_pose',observation['pose']))
            state['camera_motion_since_previous'] = dict(
                dx=round(new_pose[0]-old_pose[0],3),dy=round(new_pose[1]-old_pose[1],3),
                dyaw_rad=round(math.atan2(math.sin(new_pose[2]-old_pose[2]),math.cos(new_pose[2]-old_pose[2])),3),
                source='snapshot_pose_delta')
            number = 4 if map_images else 3
            state['image_roles']['previous_camera'] = number
            previous_images = [dict(type='text',text=f'IMAGE {number}: PREVIOUS camera (IMAGE 1 is CURRENT). Compare using recorded poses/times.'),
                               dict(type='image',image=decode_image(previous_camera['image']))]
        state["objects"] = [
            {**obj, "within_completion_range": obj.get("range_m", math.inf) <= 0.8}
            for obj in observation.get("objects", [])
        ]
        live_task = task_policy.is_task_mode(observation)
        if live_task:state["sensor_grounding"]=sensor_context(observation,camera.size)
        # Compound live instructions need useful analysis before the selector
        # reads it. Keep the simulator's existing experiment schedule intact.
        analysis_budget = max(64, self.args.analysis_tokens) if live_task else self.args.analysis_tokens
        if trajectory:
            analysis_budget = 48 if evidence else max(96, analysis_budget)
        analysis_ready = min(31 if live_task else 7, analysis_budget - 1)
        completion_eligible = live_task or any(
            obj.get("visible") and obj.get("range_m", math.inf) < 0.95
            for obj in observation.get("objects", [])
        )
        if evidence and observation.get('pipeline',{}).get('speculative'):
            completion_eligible = False
        state["valid_choices"] = labels + ([] if direct else ["W"]) + (["Z"] if completion_eligible else [])
        state["completion_rule"] = (
            (task_policy.TASK_RULES_BASE if evidence else task_policy.TASK_RULES) if live_task else
            "Z completes the task only when the requested object is visibly confirmed and its measured range is at most 0.8 m. A nearby distractor does not count."
        )
        system = DIRECT_SYSTEM if direct else SYSTEM
        if live_task:
            system = task_policy.DIRECT_SYSTEM if direct else task_policy.WAYPOINT_SYSTEM
            state["objects"] = observation.get("objects", [])
        if trajectory:
            system = task_policy.TRAJECTORY_SYSTEM
            options = state.pop('heading_options', None)
            if options is not None:
                state['available_translations'] = [
                    {'turn_actions': row.get('turn_actions', [action_plan.LABELS[letter] for letter in row['turn_prefix']]),
                     'then_forward_steps': row['full_forward_steps'],
                     'destination_map_yaw_degrees': row['map_yaw_degrees']}
                    for row in options if row['turn_sweep_known_free'] and row['full_forward_steps'] > 0]
                if evidence:
                    state['available_translations'] = translation_options(options)
                state['prefixes_without_known_forward_step'] = [row['turn_prefix'] or '(no turn)'
                    for row in options if not row['turn_sweep_known_free'] or row['full_forward_steps'] == 0]
            spec = observation.get('action_spec', {})
            state['action_geometry'] = {
                name + '_turn_deg': round(math.degrees(spec[name]['angle_rad']), 2)
                for name,item in spec.items() if 'angle_rad' in item}
        unprotected = evidence and observation.get("collision_protection") == "disabled"
        if evidence and not unprotected:
            unreliable = observation.get('recent_turn_overshoots', [])
            turn_labels = [letter for letter in turn_labels if action_plan.LABELS[letter] not in unreliable]
            state['available_translations'] = [
                row for row in state.get('available_translations', [])
                if not any(action in unreliable for action in row['turn_actions'])
            ]
        if evidence:
            system = task_policy.TRAJECTORY_EVIDENCE_SYSTEM
            if split_vision:
                system += '\n' + target_vision.RULES + (
                    '\nOnly a current target report confirms visibility. Use the separate scene '
                    'report for hazards. The prior target state is historical until current evidence '
                    'confirms it. last_seen.pose is the camera viewpoint, never an object position. '
                    'Use it spatially only within the same tracking epoch. A task with '
                    'TARGET NONE is still actionable using the instruction and measured evidence.')
                scope = observation.get('memory_context') or {}
                state['visual_anchor_epoch'] = scope.get('anchor_epoch')
                state['temporal_visual_state'] = {
                    k: v for k, v in visual_state.items() if k not in ('sightings', 'scene', 'scene_directions')
                }
                state.pop('prepared_perception', None)  # The exact-frame report is in the observer block.
                if target_images:
                    state['image_roles']['historical_target_crop'] = 4 if map_images else 3
            if observation.get('pipeline',{}).get('speculative'):system+=task_policy.PIPELINE_FORECAST_RULES
            state = dict(state)
            state["task_context"] = dict(state.get("task_context") or {})
            state["task_context"].pop("previous_analysis", None)
            state["immediate_mobility"] = immediate_mobility(
                state.get("available_translations",[]), spec, horizon)
            state["history_authority"] = "Measured poses/actions are facts; planned/discarded actions are not execution."
        if evidence:
            state['action_geometry']={code:round(math.degrees(action_plan.turn_delta(name,spec)),2) for code,name in action_plan.LABELS.items() if name in spec and 'angle_rad' in spec[name]}
            state['orientation']=orientation(observation['pose'],observation.get('maps_context') or {})
            for row in state.get('available_translations',[]):
                heading=row.pop('destination_map_yaw_degrees')
                row['body_heading_map_degrees']=heading
                reverse=row.get('translation_label','F') in ('B','C')
                row['travel_map_bearing_degrees']=round((heading+(180 if reverse else 0)+180)%360-180,2)
            first = admissible([], state.get('available_translations', []), spec, turn_labels)
            state['valid_choices'] = (labels if unprotected else first_action_labels(first, observation))
            state['valid_choices'] = state['valid_choices'] + (['Z'] if completion_eligible else [])
            if observation.get('first_motion_checks'):
                system += (
                    '\nfirst_motion_checks is the fresh raw lidar gate for the first translation. '
                    'A blocked first action or a repeated stalled action is excluded from valid_choices. '
                    'Small turns that repeatedly overshot are temporarily excluded. '
                    'Choose a different bounded manoeuvre or refresh if no movement fits. '
                    'edge_exit denotes a short verified movement out of shallow initial footprint overlap.'
                )
            state['completion_rule']=task_policy.COMMITTED_COMPLETION_RULE
            state.pop('history_authority',None)
            state.pop('prefixes_without_known_forward_step',None)
        mission = state.get('mission_context') if evidence else None
        route = (mission or {}).get('route')
        if mission:
            system += (
                '\nThe mission subgoal is a temporary, previously selected local target. '
                'Use the measured event and past attempts to avoid retracing a branch without a reason. '
                'If the checked mission route still suits the full task and current evidence, start it with its '
                'first turn, or a translation toward the target if aligned. Select another first action '
                'if current camera evidence, clearance, precise positioning or the task calls for it. '
                'Reaching a subgoal does not complete the full instruction.'
            )
        if evidence and observation.get('physical') is True and getattr(self.args,'compact_robot_prefix',True):
            fixed={key:state.pop(key) for key in ('action_spec','rover_body','action_geometry') if key in state}
            system+='\nFixed robot context (same authority as measured observation; units and action signs are literal):\n'+json.dumps(fixed,ensure_ascii=False,separators=(',',':'))
            state['robot_context_source']='system message: action_spec, rover_body, action_geometry'
        merge_observer = (prepared_observer is not None and
                          getattr(self.args,'merge_prepared_observer',False))
        observer_suffix = []
        if merge_observer:
            observer_suffix = (self.encode('<|im_start|>user\nUse the validated analysis of this exact camera frame:<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n')
                               + self.encode(prepared_observer) + self.encode('<|im_end|>\n'))
        def build_inputs(text):
            built = llm.processor.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "IMAGE 1: front camera"},
                            {"type": "image", "image": camera},
                            {"type": "text", "text": "IMAGE 2: online measured map"},
                            {"type": "image", "image": map_image},
                            *map_images,
                            *previous_images,
                            *target_images,
                            {"type": "text", "text": text},
                        ],
                    },
                ],
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            built.pop("attention_mask", None)
            return append_exact_tokens(built, observer_suffix) if observer_suffix else built

        try:
            inputs, context_budget = budgeted_inputs(
                build_inputs, state, live_task, self.input_token_limit(live_task),
                getattr(self.args, 'context_window', 8192))
        except ContextBudgetExceeded:
            if not target_images:
                raise
            # The optional identity crop may yield to current sensor evidence.
            target_images = []
            state['image_roles'].pop('historical_target_crop')
            inputs, context_budget = budgeted_inputs(
                build_inputs, state, live_task, self.input_token_limit(live_task),
                getattr(self.args, 'context_window', 8192))
            context_budget['historical_target_crop_omitted'] = True
        common, vision, reason, choice = [await llm.create_block() for _ in range(4)]
        rationale_block = await llm.create_block() if rationale_enabled else None
        vision_context = [] if merge_observer else [vision]
        common_view, prefix_lease, prefill_metrics = [common], None, {}
        events = [asyncio.Event() for _ in range(3 if rationale_enabled else 2)]
        traces = [[] for _ in events]
        timeline = []
        tasks = []

        def record(event, **extra):
            item = {"event": event, "seconds": round(time.monotonic() - started, 4), **extra}
            timeline.append(item)
            if self.rank == 0:
                print("TRACE " + json.dumps(item), flush=True)

        def prompt(text):
            return self.encode(
                "<|im_start|>user\n"
                + text
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
            )

        def decision_view():
            return common_view+vision_context+[reason,choice]

        async def stream(index, block, instruction):
            record("stream_started", stream=index)
            try:
                analysis_view = common_view + (vision_context+[reason,choice] if evidence and index==2 else vision_context if evidence and index==1 else []) + [block]
                output = await llm.forward(prompt(instruction), cache_view=analysis_view)
                budget = max(8,getattr(self.args,"spatial_comment_tokens",64)//2) if evidence and index in (1,2) else analysis_budget
                for step in range(budget):
                    token = checked_argmax(output.logits, error="Non-finite analysis logits")
                    traces[index].append(token)
                    # The action-sequence selector needs the finished spatial
                    # assessment, including execution history, before committing.
                    if step >= analysis_ready and not (trajectory and index == 1):
                        events[index].set()
                    if token == self.tokens.eos_token_id:
                        break
                    output = await llm.forward([token], cache_view=analysis_view)
            finally:
                events[index].set()
                record("stream_finished", stream=index, tokens=len(traces[index]))

        target_geometry = preference = passage = passage_goal = visual_scope = None
        try:
            record('context_prepared', **context_budget)
            record("observation_prefill_started", rank=self.rank)
            common_view, prefix_lease, prefill_metrics = await self.prefill_observation(
                inputs, system, common, live_task)
            record("observation_prefilled", tokens=int(inputs["input_ids"].numel()), **prefill_metrics)
            if evidence:
                visual_scope = await task_visual_scope(self, observation['instruction'], prompt) if split_vision else None
                if prepared_observer is not None:
                    # The exact endpoint image was analysed during execution.
                    # Import its completed report; do not decode the observer again.
                    traces[0] = self.encode(prepared_observer)
                    if not merge_observer:
                        await llm.forward(prompt('Use the validated analysis of this exact camera frame:')
                                          + traces[0] + self.encode('<|im_end|>\n'),
                                          cache_view=common_view+[vision], return_logits=False)
                    record('observer_reused', camera_stamp=observation['camera_stamp'], merged=merge_observer)
                    if split_vision:
                        visual_state = observation['prepared_perception']['visual_state']
                elif split_vision:
                    frame = target_vision.snapshot_frame(observation)

                    async def observe_forward(tokens, **kwargs):
                        return await llm.forward(tokens, cache_view=common_view+[vision], **kwargs)

                    report, frame_metrics = await inspect_frame(
                        self, camera, observation['instruction'], visual_state, visual_scope in ('A', 'T'))
                    record('frame_prefilled', **frame_metrics)
                    # This report is about IMAGE 1, even if a newer motion frame
                    # was observed. The rover checks timestamps before merging it.
                    visual_state['latest_frame'] = None
                    target_vision.apply_report(visual_state, report, frame)
                    traces[0] = self.encode(target_vision.describe(visual_state))
                    await observe_forward(prompt('Use this validated current target and scene report:') +
                                          traces[0] + self.encode('<|im_end|>\n'), return_logits=False)
                    record('target_scene_observed', target_status=report['status'])
                else:
                    await stream(0, vision, task_policy.TRAJECTORY_EVIDENCE_PROMPT)
                    await llm.forward(self.encode('<|im_end|>\n'),
                                      cache_view=common_view+[vision], return_logits=False)
                assessment = task_policy.COMMITTED_ASSESSMENT_PROMPT
                target = estimate(visual_state, observation) if visual_scope in ('A', 'T') else None
                if target and target.bearing(observation['pose'])[1] < math.pi / 2:
                    target_geometry = target.describe(observation['pose'])
                    assessment += (
                        '\nUse camera-derived target sector (positive bearing is left of the body): '
                        + json.dumps(target_geometry, separators=(',', ':'))
                        + '. Treat a historical sector as a clue for regaining visual contact. '
                        'Use the sector only if it is relevant to the current requirement. '
                        'Allow checked detours. This estimate cannot verify measured range or arrival.'
                    )
                    if visual_scope == 'A':
                        preference = TrajectoryPreference(target, observation, state['available_translations'],
                                                          state['valid_choices'], turn_labels)
                if visual_scope == 'M' and mission:
                    passage, passage_goal = choose_passage(observation, visual_state.get('scene_directions', {}))
                    if passage:
                        assessment += (
                            '\nThe measured passable route for this observation is: '
                            + json.dumps({k: passage[k] for k in ('turn_actions', 'translation_label',
                                                                 'then_forward_steps')})
                            + '. Use it to advance towards open floor. '
                            'A short recovery move is followed by reassessment, not continued blind retreat.'
                        )
                        record('passage_selected', goal=passage_goal, route=passage)
                await stream(1,reason,assessment)
                await llm.forward(self.encode('<|im_end|>\n'),cache_view=common_view+vision_context+[reason],return_logits=False)
            else:
                tasks = [
                    asyncio.create_task(
                        stream(
                            0,
                            vision,
                            (task_policy.TRAJECTORY_EVIDENCE_PROMPT if evidence else task_policy.OBSERVER_PROMPT if live_task else
                             "Describe the actual RGB camera image briefly: visible object colors/shapes, target evidence, and occlusion. "
                             "Do not infer target visibility from the task. Maximum two short sentences."),
                        )
                    ),
                    asyncio.create_task(
                        stream(
                            1,
                            reason,
                            task_policy.TRAJECTORY_SPATIAL_PROMPT if trajectory else task_policy.SPATIAL_PROMPT if live_task else (
                                "First check completion from actual RGB and measured range. Otherwise reason about current heading, passages, unknown space and action history. Choose F, L, R or S to find the target; escape loops and blocked directions. Maximum two short sentences."
                                if direct
                                else "First check task completion: if the requested object is visibly nearby within the instructed range, recommend Z and no further movement. Otherwise "
                                "briefly compare candidate viewpoints for the task using map geometry, unexplored space, path length, "
                                "visits and failures. Suggest a promising direction. Maximum two short sentences."
                            ),
                        )
                    ),
                ]
                await asyncio.gather(*(event.wait() for event in events))
            record(
                "selector_started",
                vision_tokens=len(traces[0]),
                reason_tokens=len(traces[1]),
                analysis_still_running=any(not task.done() for task in tasks),
            )
            # Live completion evaluates the full instruction. Simulation retains
            # its measured-object completion and independent evaluation rules.
            if evidence:
                allowed = [label for label in state['valid_choices'] if label != 'Z']
                if passage:
                    allowed = passage_labels([], passage, spec, allowed)
            else:
                allowed = labels if direct else labels + ["W"]
            completion_scores = None
            completion_top_tokens = []
            selected = None
            selected_token = None
            action_token_support = []
            choice_prefix = []
            if evidence and observation.get('pipeline',{}).get('speculative'):
                output = await llm.forward(prompt(
                    'FORECAST CONTINUATION: Assume all committed_remaining_actions succeed. '
                    'At the conditional task_context/planned_start_pose, is ADDITIONAL locomotion '
                    'justified for a remaining instruction requirement? Answer No if pending work '
                    'would satisfy the requested movement, or the next step needs fresh measured '
                    'verification; choose S at the boundary in that case. Do not repeat pending '
                    'work. This is not a claim of completion now. Answer exactly Yes or No:'),
                    cache_view=decision_view())
                if not torch.isfinite(output.logits.float()).all():
                    raise FloatingPointError('Non-finite forecast decision logits')
                forecast_probs, forecast_tokens = binary_probabilities(output.logits.float(),self.tokens)
                record('forecast_continuation_checked', probabilities=dict(zip(('yes','no'),forecast_probs)))
                if forecast_probs[1] >= forecast_probs[0]:
                    selected='S';allowed=['S'];scores=[1.0]
                else:
                    choice_prefix=[forecast_tokens[0]]+self.encode('<|im_end|>\n')
            task_reminder=''
            if evidence and getattr(self.args,'task_reminder',True):
                literal=observation.get('instruction','')
                if len(self.encode(literal))<=256:
                    task_reminder='Literal current user instruction: '+literal+'\n'
            if completion_eligible:
                # Completion is a separate semantic question. It must not lose
                # to a movement label merely because the prompt asks for a point.
                completion_prompt = (
                    task_policy.COMPLETION_PROMPT if live_task else
                    "Before choosing any movement: is the search task already complete? "
                    "Answer yes ONLY if the requested object is clearly visible in IMAGE 1 "
                    "and a matching object proposal has within_completion_range=true. "
                    "A nearby object of another color/class is not the target. "
                    "Use actual RGB and the measured object facts. Answer exactly Yes or No:"
                )
                output = await llm.forward(
                    prompt(task_reminder+completion_prompt),
                    cache_view=decision_view() if live_task else common_view + [choice],
                )
                binary = output.logits.float()
                if not torch.isfinite(binary).all():
                    raise FloatingPointError("Non-finite completion logits")
                probabilities, binary_tokens = binary_probabilities(binary, self.tokens)
                top = binary.topk(8)
                completion_top_tokens = [
                    {"text": self.tokens.decode([int(token)]), "logit": round(float(value), 3)}
                    for token, value in zip(top.indices.cpu(), top.values.cpu())
                ]
                completion_scores = dict(zip(("yes", "no"), probabilities))
                record("completion_checked", probabilities=completion_scores)
                if probabilities[0] > probabilities[1]:
                    selected = "Z"
                    allowed = ["Z", "W"]
                    scores = probabilities
                else:
                    choice_prefix = [binary_tokens[1]] + self.encode("<|im_end|>\n")
            instruction = (
                "Choose the available action that advances the next unfinished part of the user's instruction. "
                + ("F=forward, L=turn left, R=turn right, S=stop or wait. Reply with one letter F, L, R or S:"
                   if direct else "Reply with one letter: " + ", ".join(allowed) + ". W means wait if no candidate is suitable:")
                if live_task else (
                "Choose the next driving action: F=forward, L=turn left, R=turn right, S=stop. "
                "Use the CURRENT heading, observed free space and past blocked actions. No waypoint planner will help. "
                "Reply with exactly one letter F, L, R or S:"
                if direct
                else "Choose the best next candidate for the task. Reply with exactly one letter: "
                + ", ".join(allowed)
                + ". W means wait if none is suitable. "
                "Camera detections may be distractors. Your single letter is:"
                )
            )
            if trajectory:
                instruction = (
                    f"Generate a sequence of at most {horizon} actions for the next unfinished requirement. " +
                    ("Select ONE authoritative plan from current camera, orientation, admissible actions and measured history. " if evidence else "Use the completed spatial assessment and maneuver_memory; preserve executed turn progress. ") +
                    "Read action_geometry for actual turn angles. Do not undo recent turns into a blocked heading "
                    "without a measured reason or a requirement in the user's instruction. " +
                    ("Use action_spec and valid_choices. F=forward10cm, G=forward5cm, B=reverse10cm, C=reverse5cm when provided; Q/E=left/right11.25deg, L/R=left/right22.5deg, A/D=left/right45deg; S refresh. These are nominal targets; measured execution is authoritative. " if evidence else "Use F=forward, L=left, R=right, S=stop and refresh observations. ") +
                    "Update the predicted heading after each turn. All forward segments must fit " +
                    ("observed free space. Plan useful progress up to the next task or clearance boundary, "
                     "using normal steps in open space and short steps near that boundary. "
                     "S requires a concrete reason to refresh now, not merely the end of one primitive. "
                     if evidence else "observed free space. End early with S if more evidence is needed. ") +
                    "Output only comma-separated action letters. First action:"
                )
            instruction=task_reminder+instruction
            selected_route = None
            if selected is None:
                output = await llm.forward(
                    choice_prefix + prompt(instruction), cache_view=decision_view()
                )
                if evidence:
                    distribution=self.label_distribution(output.logits,allowed,error="Non-finite selector logits")
                    scores=distribution.probabilities
                    index = (preference.choose((), allowed, scores) if preference
                             else max(range(len(scores)), key=scores.__getitem__))
                    selected_token=distribution.token_ids[index]
                    action_token_support.append(distribution.valid_probability_mass)
                else:
                    scores = checked_probabilities(output.logits,
                        [one_token(self.tokens, label) for label in allowed],
                        error="Non-finite selector logits")
                    index=max(range(len(scores)),key=scores.__getitem__)
                selected=allowed[index]
            if passage and selected != "Z":
                selected_route = passage
            elif route and visual_scope != 'M' and not unprotected and selected not in ("S", "Z"):
                # The current selector can decline the proposal with its first action.
                # Once it accepts the first primitive, preserve that branch for this plan.
                if selected in route_labels([], route, spec):
                    selected_route = route
            plan_labels = []
            step_probabilities = []
            if trajectory and selected != "Z":
                plan_labels.append(selected)
                step_probabilities.append(dict(zip(allowed, scores)))
                while len(plan_labels) < horizon and plan_labels[-1] != "S":
                    output = await llm.forward(
                        [selected_token if selected_token is not None else one_token(self.tokens, plan_labels[-1])] + self.encode(","),
                        cache_view=decision_view(),
                    )
                    labels = ([k for k,v in action_plan.LABELS.items() if v in observation.get("action_spec",{})] if unprotected else admissible(plan_labels,state.get("available_translations",[]),spec,turn_labels) if evidence else ["F","L","R","S"])
                    if passage:
                        labels = passage_labels(plan_labels, passage, spec, labels)
                    elif selected_route is not None:
                        on_route = route_labels(plan_labels, selected_route, spec)
                        labels = [label for label in labels if label in on_route]
                    if evidence:
                        distribution=self.label_distribution(output.logits,labels,error="Non-finite trajectory logits")
                        probabilities=distribution.probabilities
                        index = (preference.choose(plan_labels, labels, probabilities)
                                 if preference and selected_route is None
                                 else max(range(len(probabilities)), key=probabilities.__getitem__))
                        selected_token=distribution.token_ids[index]
                        action_token_support.append(distribution.valid_probability_mass)
                    else:
                        probabilities = checked_probabilities(output.logits,
                            [one_token(self.tokens, label) for label in labels],
                            error="Non-finite trajectory logits")
                        index=max(range(len(probabilities)),key=probabilities.__getitem__)
                    plan_labels.append(labels[index])
                    step_probabilities.append(dict(zip(labels, probabilities)))
                record("trajectory_generated", actions=plan_labels)
            selected_plan=None
            rationale_consistency=None
            if rationale_enabled:
                # The last selected token has not yet been fed back to the cache.
                # Commit it before the rationale so every action, including S,
                # is actually present in the block the rationale reads.
                if selected=='Z':tail=[binary_tokens[0]]+self.encode('<|im_end|>\n')
                elif selected_token is None:tail=[forecast_tokens[1]]+self.encode('<|im_end|>\n')
                else:tail=[selected_token]+self.encode('<|im_end|>\n')
                await llm.forward(tail,cache_view=decision_view(),return_logits=False)
            if evidence:
                committed_actions=['stop'] if selected=='Z' else [action_plan.LABELS[c] for c in plan_labels]
                selected_plan=make_plan(committed_actions,observation['pose'],spec,unprotected,selected=='Z')
                selected_plan.update(snapshot_id=observation['snapshot_id'],map_version=observation['map_version'])
                record('plan_committed',actions=committed_actions)
            record("selector_finished", selected=selected)
            def reasoner_text():
                if evidence and not rationale_enabled:
                    return 'Plan: '+selected_plan['display']
                text=self.tokens.decode(traces[2] if evidence else traces[1],skip_special_tokens=True)
                if evidence and not traces[2]:text='Explanation is being prepared.'
                if rationale_consistency is not None and not rationale_consistency['consistent']:
                    text='The generated explanation was withheld because it did not describe the committed plan.'
                return ('Plan: '+selected_plan['display']+'\nRationale: '+text) if selected_plan is not None else text
            response = {
                "goal_id": selected,
                "action_token_support": action_token_support,
                "map_version": observation["map_version"],
                "snapshot_id": observation["snapshot_id"],
                "episode_id": observation.get("episode_id"),
                "probabilities": dict(zip(allowed, scores)),
                "completion_probabilities": completion_scores,
                "completion_mode": task_policy.MODE if live_task else "measured_range",
                "completion_top_tokens": completion_top_tokens,
                "observer": self.tokens.decode(traces[0], skip_special_tokens=True),
                "reasoner": reasoner_text(),
                "observer_reused": prepared_observer is not None,
                "planning_mode": "committed_spatial_plan_v1" if evidence else "legacy",
                "observer_budget_exhausted": bool(evidence and not split_vision and len(traces[0]) == analysis_budget),
                "spatial_budget_exhausted": False,
                "assessment_budget_exhausted": bool(evidence and len(traces[1])==max(8,getattr(self.args,"spatial_comment_tokens",64)//2) and traces[1][-1]!=self.tokens.eos_token_id),
                "latency_s": time.monotonic() - started,
                "timeline": list(timeline),
                "analysis_in_progress": any(not task.done() for task in tasks),
                "image_sha256": {
                    "camera": hashlib.sha256(camera.tobytes()).hexdigest(),
                    "map": hashlib.sha256(map_image.tobytes()).hexdigest(),
                    **({'accumulated_map': hashlib.sha256(map_images[-1]['image'].tobytes()).hexdigest()} if map_images else {}),
                },
                "input_tokens": int(inputs["input_ids"].numel()),
                "context_budget": context_budget,
                "prefill": prefill_metrics,
                "metadata": self.metadata,
                "runtime_metrics": {k:v-metrics_before.get(k,0) for k,v in
                    {**engine_stats,**cache_statistics(self)}.items()},
            }
            if mission:
                response['mission_context'] = {**mission, 'route_accepted': selected_route is not None}
            if split_vision:
                response['target_visual_state'] = visual_state
                response['visual_task_scope'] = visual_scope
                if passage and selected != 'Z':
                    response['passage_goal'] = passage_goal
                if target_geometry is not None:
                    response['target_geometry'] = target_geometry
            if selected_plan is not None:
                response.update(selected_plan=selected_plan,rationale_consistency=rationale_consistency,
                                rationale_enabled=rationale_enabled,
                                rationale_pending=False,
                                raw_spatial_rationale=self.tokens.decode(traces[2],skip_special_tokens=True) if rationale_enabled else '',
                                spatial_assessment=self.tokens.decode(traces[1],skip_special_tokens=True),
                                spatial_rationale=reasoner_text().split('\nRationale: ',1)[-1] if rationale_enabled else '',
                                reasoner_role='rationale_for_committed_plan' if rationale_enabled else 'selected_plan')
            if live_task:
                response['task_complete'] = selected == 'Z'
            if direct:
                response.update(
                    protocol="direct_navigation_v1",
                    action=({**action_plan.LABELS,"Z":"stop"} if trajectory else {"F":"forward","L":"left","R":"right","S":"stop","Z":"stop"})[selected],
                    task_complete=selected == "Z",
                )
            if trajectory:
                response.update(
                    protocol=action_plan.PROTOCOL,
                    trajectory_steps=horizon,
                    actions=["stop"] if selected == "Z" else [action_plan.LABELS[label] for label in plan_labels],
                    step_probabilities=step_probabilities,
                )
                action_plan.validate_response(response, observation)
            early = rationale_enabled and selected != 'Z' and observation.get('response_mode') == 'plan_stream_v1'
            if early:
                response.update(message_kind='plan_ready', rationale_pending=True)
                if on_selection is not None:on_selection(dict(response))
            if rationale_enabled:
                rationale=task_policy.COMMITTED_RATIONALE_PROMPT+'\nCommitted plan: '+json.dumps(
                    {'actions':committed_actions,'task_complete':selected=='Z'},separators=(',',':'))
                await stream(2,rationale_block,rationale)
                await llm.forward(self.encode('<|im_end|>\n'),cache_view=common_view+vision_context+[reason,choice,rationale_block],return_logits=False)
                review=await llm.forward(prompt(task_policy.RATIONALE_CONSISTENCY_PROMPT),
                    cache_view=common_view+vision_context+[reason,choice,rationale_block])
                review_scores,_=binary_probabilities(review.logits.float(),self.tokens)
                rationale_consistency=dict(consistent=review_scores[0]>=review_scores[1],
                    probabilities=dict(zip(('yes','no'),review_scores)))
                record('rationale_checked',**rationale_consistency)
                response.update(reasoner=reasoner_text(), rationale_consistency=rationale_consistency,
                    raw_spatial_rationale=self.tokens.decode(traces[2],skip_special_tokens=True),
                    spatial_rationale=reasoner_text().split('\nRationale: ',1)[-1],
                    spatial_budget_exhausted=bool(len(traces[2])==max(8,getattr(self.args,'spatial_comment_tokens',64)//2) and traces[2][-1]!=self.tokens.eos_token_id),
                    timeline=list(timeline))
                if early:response.update(message_kind='rationale_ready',rationale_pending=False)
                else:response['latency_s']=time.monotonic()-started
            # Synchronous socket write: publish the chosen goal immediately,
            # without adding rank-dependent awaits to the TP coroutine schedule.
            # Analysis continues while ROS validates/executes the waypoint.
            if on_selection is not None and not early:
                on_selection(response)
            await asyncio.gather(*tasks)
            peak = torch.cuda.max_memory_allocated(self.rank) / 2**20
            peaks = [None] * self.world
            dist.all_gather_object(peaks, round(peak, 1), group=llm.engine.tp_cpu_group)
            response.update(
                observer=self.tokens.decode(traces[0], skip_special_tokens=True),
                reasoner=reasoner_text(),
                timeline=timeline,
                analysis_in_progress=False,
                analysis_completed_latency_s=time.monotonic() - started,
                gpu_peak_allocated_mib=peaks,
                runtime_metrics={k:v-metrics_before.get(k,0) for k,v in {**engine_stats,**cache_statistics(self)}.items()},
            )
            if self.rank == 0:
                print("INFERENCE_AUDIT " + json.dumps(response), flush=True)
            return response
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for block in (common, vision, reason, choice,rationale_block):
                if block is not None:await llm.free_block(block)
            if prefix_lease is not None:
                await self.system_prefix_cache.release(prefix_lease)


async def run(args):
    from robot_sim.candidate_nav.mission_context import PROTOCOLS as MISSION_PROTOCOLS

    torch.set_num_threads(4 if getattr(args,'hardware_profile','legacy')=='h100' else 1)
    policy = AsyncCandidatePolicy(args)
    policy.metadata['persistent_memory']='cpu_episode_v1'
    memory=None
    if not policy.rank:
        from pathlib import Path
        directory=os.environ.get('ROBOT_LLM_MEMORY_DIR') or str(Path(__file__).resolve().parents[1]/'.runtime/live_rover/memory')
        memory=MemoryCoordinator(directory,lambda text:len(policy.tokens.encode(text,add_special_tokens=False)))
    print("MODEL_METADATA " + json.dumps(policy.metadata), flush=True)

    async def evaluate(payload, on_selection=None):
        try:
            if payload.get("protocol") in MISSION_PROTOCOLS:
                from .mission_policy import evaluate_mission
                return await evaluate_mission(policy, payload)
            if payload.get("protocol") == "motion_visual_memory_v1":
                from .motion_vision_policy import inspect
                return await inspect(policy, payload)
            if payload.get("protocol") == "candidate_generation_v1":
                return await policy.propose(payload)
            return await policy.decide(payload, on_selection)
        except Exception as error:
            import traceback

            traceback.print_exc()
            from .context_budget import error_payload
            return error_payload(error)

    try:
        if policy.rank:
            while True:
                packet = [None]
                dist.broadcast_object_list(packet, src=0, group=policy.llm.engine.tp_cpu_group)
                if packet[0] is None:
                    break
                if packet[0].get("keepalive"):
                    continue
                await evaluate(packet[0])
            return
        lock = asyncio.Lock()

        async def keepalive():
            # Rank 1 waits in a timed Gloo broadcast even when no client is
            # connected. Keep that rendezvous alive between episodes.
            while True:
                await asyncio.sleep(30)
                async with lock:
                    if policy.world > 1:
                        dist.broadcast_object_list(
                            [{"keepalive": True}], src=0, group=policy.llm.engine.tp_cpu_group
                        )

        async def handle_message(reader, writer, image_cache):
            sent = False
            payload={}
            memory_status=None
            received_at=time.monotonic()
            admission={}

            def publish(response):
                nonlocal sent
                if memory_status is not None:response['memory_status']=memory_status
                response['transport_keepalive']=payload.get('_transport_v')==2
                response['server_timing']={**admission,'received_to_publish_s':time.monotonic()-received_at}
                encoded=(json.dumps(response,allow_nan=False)+'\n').encode()
                try:writer.write(encoded)
                except (OSError,ConnectionError,RuntimeError):pass
                # Socket failure must not alter the TP forward schedule. Both
                # ranks finish/free the same blocks even after the client leaves.
                sent = True

            try:
                payload = image_cache.decode(json.loads(await asyncio.wait_for(reader.readline(), 300)))
                if payload.get("health"):
                    response = {"ready": True, **policy.metadata}
                else:
                    queued_at=time.monotonic()
                    async with lock:
                        admission["queue_s"]=time.monotonic()-queued_at
                        original=payload
                        payload=await memory.prepare(payload)
                        memory_status=payload.get('_memory_status')
                        admission['memory_prepare_ms']=(memory_status or {}).get('prepare_ms')
                        if policy.world > 1:
                            dist.broadcast_object_list(
                                [payload], src=0, group=policy.llm.engine.tp_cpu_group
                            )
                        response = await evaluate(payload, publish)
                        await memory.record(original,response)
                if sent and payload.get('response_mode')=='plan_stream_v1' and response.get('error'):
                    publish(dict(message_kind='rationale_error',snapshot_id=payload['snapshot_id'],map_version=payload['map_version'],error=response['error']))
                elif not sent or payload.get('response_mode')=='plan_stream_v1' and response.get('message_kind')=='rationale_ready':
                    publish(response)
                await writer.drain()
            except Exception as error:
                if not sent:
                    try:
                        publish({"error": str(error), "goal_id": "W"})
                        await writer.drain()
                    except ConnectionError:
                        pass
            return sent and not writer.is_closing() and payload.get('_transport_v')==2

        async def handle(reader, writer):
            from robot_sim.candidate_nav.wire_images import WireImages
            image_cache=WireImages()
            try:
                while not reader.at_eof():
                    if not await handle_message(reader,writer,image_cache):break
            finally:
                writer.close()
                try:await writer.wait_closed()
                except ConnectionError:pass

        server = await asyncio.start_server(handle, "127.0.0.1", args.port, limit=5_000_000)
        print(f"READY 127.0.0.1:{args.port}", flush=True)
        heartbeat = asyncio.create_task(keepalive())
        try:
            async with server:
                await server.serve_forever()
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
    finally:
        if memory is not None:await memory.close()
        await policy.close()


def main():
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1)
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=18766)
    p.add_argument("--quantization", choices=["none", "fp8"], default="fp8")
    p.add_argument(
        "--distributed-addr", default="env://" if "RANK" in os.environ else "tcp://127.0.0.1:2339"
    )
    p.add_argument("--hardware-profile", choices=("legacy", "h100"), default="legacy")
    p.add_argument("--analysis-tokens", type=int, default=24)
    p.add_argument("--spatial-comment-tokens", type=int, choices=range(24,65), default=64)
    p.add_argument("--trajectory-policy", choices=("legacy", "evidence"), default="legacy")
    p.add_argument("--context-window", type=int, choices=(8192, 12288), default=8192,
                   help="Total engine context; live launcher uses 12288, simulation keeps 8192")
    p.add_argument("--no-system-prefix-cache", action="store_true",
                   help="Disable live system-prefix reuse for controlled comparisons")
    p.add_argument('--decode-cuda-graphs',action=argparse.BooleanOptionalAction,default=None,
                   help='Decode graphs: enabled by default on H100; --no-decode-cuda-graphs disables them')
    p.add_argument('--merge-prepared-observer',action='store_true',
                   help='Experimental exact-token merge of an already analysed camera report')
    p.add_argument('--compact-robot-prefix',action=argparse.BooleanOptionalAction,default=True,
                   help='Cache fixed robot parameters with the system prefix')
    p.add_argument('--prefill-cuda-graphs',action=argparse.BooleanOptionalAction,default=None,
                   help='Short prefill graphs: enabled for H100 live 12K; --no-prefill-cuda-graphs disables')
    p.add_argument('--prefill-rows',type=int,
                   help='Experimental prefill row budget; default uses the hardware profile')
    args = p.parse_args()
    if not 8 <= args.analysis_tokens <= 128:
        p.error("analysis-tokens must be 8..128")
    if args.prefill_rows is not None and (args.prefill_rows<64 or args.prefill_rows%64):
        p.error('prefill-rows must be a positive multiple of 64, at least 64')
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
