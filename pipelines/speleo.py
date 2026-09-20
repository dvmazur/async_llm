"""One append-only conversation per episode; roles generate strictly in sequence."""
from dataclasses import dataclass
import json
import time

from experiment_runner.logs import atomic
from experiment_runner.blocks import BlockHandle
from experiment_runner.generation import Generation
from .prompts import SPELEO, role_request, action_request, history_event


@dataclass(frozen=True)
class RoleParams:
    budget: int
    temperature: float
    seed_offset: int
    top_k: int = 20
    top_p: float = .9


ROLE_PARAMS = {
    'observer': RoleParams(18, .35, 1),
    'planner': RoleParams(60, .65, 2),
    'executor': RoleParams(16, .45, 5),
}
ROLE_ORDER = ('observer', 'planner', 'executor')


def position_info(info):
    position = (info or {}).get('player_pos')
    if position is None:
        return {}, None
    values = [float(v) for v in position]
    return {'player_pos': values}, values[1]


class SpeleoPipeline:
    def __init__(self, world, recorder, engine, *, context, max_actions=100,
                 role_params=None, planner_interval=10):
        if type(planner_interval) is not int or planner_interval < 1:
            raise ValueError('planner_interval must be a positive integer')
        self.world, self.recorder, self.engine = world, recorder, engine
        self.context, self.max_actions = context, max_actions
        self.params = {**ROLE_PARAMS, **(role_params or {})}
        self.planner_interval = planner_interval
        self.stop_requested = lambda: False
        self.common = self.history = None

    async def generate_tokens(self, role, step, rng, *, last_action, close_previous):
        """Append this role's prompt and every nonterminal token to the same block."""
        params = self.params[role]
        result, error = Generation(), None
        started = time.perf_counter()
        try:
            await self.engine.generate(
                role_request(role, step, last_action, close_previous=close_previous),
                [self.common], self.history, result=result, generator=rng, budget=params.budget,
                temperature=params.temperature, top_k=params.top_k, top_p=params.top_p)
            return result.text
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            self.recorder.log('stream', dict(role=role, observation=step, text=result.text,
                sampled_tokens=result.sampled_tokens, visible_tokens=result.visible_tokens,
                seconds=time.perf_counter()-started, live_dependencies_at_start=[], error=error))

    async def rollout(self, initial, rngs):
        previous = current = initial.image
        last_action, feedback = 'none', None
        plan_text, plan_step = 'No published subgoal yet; use the overall task goal.', -1
        action_names = [name for name, _ in SPELEO.actions]
        action_ids = [self.engine.encode(name) for name in action_names]
        if not all(len(ids) == 1 for ids in action_ids):
            raise ValueError('Speleo action readout requires single-token action names')
        action_ids = [ids[0] for ids in action_ids]

        for step in range(self.max_actions):
            if self.stop_requested():
                break
            started = time.perf_counter()
            # The preceding assistant turn was closed after its chosen action.
            # Images and feedback are appended, not used to rebuild a fresh context.
            images = [dict(role='user', content=[
                dict(type='text', text=(
                    f'Observation {step}. Last action: {last_action}. First image previous; second image current.\n'
                    f'Feedback from the previous action: {json.dumps(feedback, ensure_ascii=False)}')),
                dict(type='image', image=previous), dict(type='image', image=current)])]
            await self.engine.prefill_messages(images, [self.common], self.history)
            answers = {}
            new_plan = step % self.planner_interval == 0
            for role in ROLE_ORDER:
                if role == 'planner' and not new_plan:
                    continue  # the previous plan is already in the growing conversation
                answers[role] = await self.generate_tokens(role, step, rngs[role],
                    last_action=last_action, close_previous=role != 'observer')
                if role == 'planner' and answers[role].strip():
                    plan_text, plan_step = answers[role].strip(), step

            output = await self.engine.prefill(action_request(SPELEO, step),
                                               [self.common], self.history)
            scores = await self.engine.score_tokens(output, action_ids)
            index = max(range(len(scores)), key=scores.__getitem__)
            action = action_names[index]
            # Scoring alone does not write the chosen token to KV. Retain the
            # actual action, then close this assistant turn before the next image.
            await self.engine.decode(action_ids[index], [self.common], self.history)
            await self.engine.prefill('<|im_end|>\n', [self.common], self.history)
            latency = time.perf_counter()-started
            self.recorder.log('decision', dict(observation=step, action_index=index, action=action,
                action_probabilities=dict(zip(action_names, scores)), decision_seconds=latency,
                draft='', assessment=answers['executor'],
                published_plan=plan_text, published_plan_based_on=plan_step,
                new_planner_started=new_plan, requested_replan=False))
            # Evaluation log only: no second history block or summarized context.
            self.recorder.log('event', history_event(step, last_action, answers['observer'], feedback))

            before = time.perf_counter()
            observation = await self.world.pass_action(index)
            seconds = time.perf_counter()-before
            info, height = position_info(observation.info)
            await self.recorder.step(step=step+1, image=observation.image, action=action,
                action_index=index, reward=observation.reward, done=observation.done,
                info=info, height=height, decision_seconds=latency, world_seconds=seconds)
            print('ACTION', json.dumps(dict(episode=self.context.episode_id,
                step=step+1, height=height, action=action)), flush=True)
            last_action = action
            previous, current = current, observation.image
            feedback = dict(reward=observation.reward, episode_done=observation.done)
            if observation.done:
                break

    async def run(self):
        start = end = None
        status = 'failed'
        try:
            await self.recorder.event('reset_started')
            initial = await self.world.reset()
            if getattr(self.world, 'metadata', None):
                self.recorder.log('world_metadata', self.world.metadata)
            info, height = position_info(initial.info)
            await self.recorder.observation(step=0, image=initial.image, info=info, height=height)
            await self.recorder.event('reset_finished')
            self.common = await self.engine.cached_prefix(SPELEO.system())
            start = time.monotonic()
            await self.recorder.event('workload_started')
            self.history = await BlockHandle.create(self.engine, f'{self.context.world_seed}/conversation')
            rngs = {}
            for role, params in self.params.items():
                seed = 1_000_003 * (self.context.model_seed+1) + params.seed_offset
                rngs[role] = self.engine.new_generator(seed)
                self.recorder.log('role_parameters', dict(role=role,
                    seed=seed if rngs[role] is not None else None,
                    temperature=params.temperature, top_k=params.top_k, top_p=params.top_p, budget=params.budget))
            await self.rollout(initial, rngs)
            status = 'stopped' if self.stop_requested() else 'completed'
        finally:
            try:
                try:
                    if self.history is not None:
                        await self.history.aclose()
                        self.history = None
                finally:
                    if self.common is not None:
                        await self.common.aclose()
                        self.common = None
                end = time.monotonic()
                await self.recorder.event('workload_finished', status=status)
            except BaseException:
                status = 'failed'
                raise
            finally:
                try:
                    await self.world.aclose()
                except BaseException:
                    status = 'failed'
                    raise
                finally:
                    try:
                        await self.recorder.finish()
                    except BaseException:
                        status = 'failed'
                        raise
                    finally:
                        atomic(self.context.results_directory/'completion.json', dict(status=status,
                            workload_start=start, workload_end=end, completed_monotonic=time.monotonic()))


def make_pipeline(engine, context):
    from experiment_runner import Recorder
    from .world import SpeleoWorld
    return SpeleoPipeline(SpeleoWorld(seed=context.world_seed),
        Recorder(context.results_directory, dump_images=False, gif_on=False), engine, context=context)
