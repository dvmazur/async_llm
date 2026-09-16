"""Speleo's entire coroutine protocol: three background roles and one executor.

Roles receive observations/decisions, not preassembled generation jobs. They choose
their own prompts and KV inputs; Generation.blocks exposes live answers. The executor
owns World interaction and has sequential draft/refine phases. Engine only batches
the resulting model requests. Prompts are text data in prompts.py.
"""
import asyncio
from dataclasses import dataclass, field
import json
import time

from experiment_runner.logs import atomic
from experiment_runner.blocks import BlockHandle
from .prompts import SPELEO, role_request, action_request, message
from .prompts import recent_text, history_event, event_text


@dataclass(frozen=True)
class RoleParams:
    budget: int
    temperature: float
    seed_offset: int
    top_k: int = 20
    top_p: float = .9


ROLE_PARAMS = {
    'observer': RoleParams(48, .35, 1),
    'planner': RoleParams(112, .65, 2),
    'executor_draft': RoleParams(28, .6, 3),
    'falsifier': RoleParams(40, .45, 4),
    'executor_refine': RoleParams(28, .45, 5),
}


@dataclass(eq=False)
class Generation:
    """One request/reply shared by coroutines, including its growing KV answer."""
    phase: str
    step: int
    reads: list
    prefix: object
    tail: object
    prompt: str
    text: str = ''
    tokens: list = field(default_factory=list)
    sampled_tokens: int = 0
    done: bool = False
    closed: bool = False
    error: BaseException | None = None
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def blocks(self):
        return [self.prefix, self.tail]

    async def wait_tokens(self, count):
        async with self.changed:
            await self.changed.wait_for(lambda: len(self.tokens) >= count or self.done)
        if self.error:
            raise self.error

    async def finish(self, *, raise_error=True):
        async with self.changed:
            await self.changed.wait_for(lambda: self.done)
        if raise_error and self.error:
            raise self.error
        return self.text


@dataclass
class Observe:
    step: int
    previous: object
    current: object
    last_action: str
    image: object


@dataclass
class PlanBoundary:
    step: int
    replan: bool


@dataclass
class PlanObservation:
    step: int
    image: object
    observer: Generation


@dataclass
class Review:
    step: int
    recent: object
    image: object
    observer: Generation
    plan: Generation | None
    draft: Generation
    new_plan: bool


def position_info(info):
    position = (info or {}).get('player_pos')
    if position is None:
        return {}, None
    values = [float(v) for v in position]
    return {'player_pos': values}, values[1]


class SpeleoPipeline:
    def __init__(self, world, recorder, engine, *, context, max_actions=100,
                 role_params=None, planner_interval=6):
        self.world, self.recorder, self.engine = world, recorder, engine
        self.context, self.max_actions = context, max_actions
        self.params = {**ROLE_PARAMS, **(role_params or {})}
        self.planner_interval = planner_interval
        self.stop_requested = lambda: False
        self.observations = asyncio.Queue()
        self.plans = asyncio.Queue()
        self.objections = asyncio.Queue()
        self.jobs = set()
        self.actors = []
        self.error = None
        self.common = self.history = None
        self.events = []

    async def ask(self, inbox, message):
        """The sender keeps inputs alive until the role acknowledges ownership."""
        if self.error:
            raise self.error
        reply = asyncio.get_running_loop().create_future()
        inbox.put_nowait((message, reply))
        return await reply

    async def observer(self, rng):
        """Compare the two frames and expose a growing description to other roles."""
        while (request := await self.observations.get()) is not None:
            observation, reply = request
            try:
                images = [dict(role='user', content=[
                    dict(type='text', text=f'Observation {observation.step}. Last action: {observation.last_action}. First image previous; second image current.'),
                    dict(type='image', image=observation.previous),
                    dict(type='image', image=observation.current)])]
                await self.engine.prefill_messages(images, [self.common], observation.image)
                job = await self.request('observer', observation.step,
                    [self.common, observation.image], last_action=observation.last_action,
                    close_previous=False)
                reply.set_result(job)
                await self.generate_tokens(job, self.params['observer'], rng)
            except BaseException as exc:
                self.error = self.error or exc
                if not reply.done():
                    reply.set_exception(exc)
                raise

    async def planner(self, rng):
        """Own plan lifetime, refresh policy, history snapshots and publication.

        Token generation can span several world actions. Its child task does only
        model IO, so this coroutine can answer new observations while it is pending.
        Completed plans become visible at action boundaries, never mid-decision.
        """
        job = generating = None
        text, based_on = 'No published subgoal yet; use the overall task goal.', -1
        replan = False
        try:
            while (request := await self.plans.get()) is not None:
                message, reply = request
                try:
                    if isinstance(message, PlanBoundary):
                        replan = replan or message.replan
                        if job is not None and job.done:
                            answer = await job.finish()
                            if answer.strip():
                                text, based_on = answer.strip(), job.step
                                self.recorder.log('plan_published', dict(current_observation=message.step,
                                    based_on=based_on, age_in_actions=message.step-based_on, text=text))
                            await generating
                            await self.close_generation(job)
                            job = generating = None
                        reply.set_result((text, based_on, replan))
                    elif isinstance(message, PlanObservation):
                        start = job is None and (based_on < 0 or replan
                            or message.step-based_on >= self.planner_interval)
                        if start:
                            frozen = await self.history.snapshot(
                                f'{self.context.world_seed}/{message.step}/history_snapshot')
                            try:
                                self.recorder.log('history_snapshot', dict(observation=message.step,
                                    tokens=frozen.raw.num_tokens, total_events=len(self.events)))
                                job = await self.request('planner', message.step,
                                    [self.common, frozen, message.image, *message.observer.blocks])
                            finally:
                                await frozen.aclose()
                            generating = asyncio.create_task(
                                self.generate_tokens(job, self.params['planner'], rng),
                                name=f'{self.context.episode_id}/planner/tokens')
                            replan = False
                        reply.set_result((job, start))
                    else:
                        raise TypeError(f'Unexpected planner message: {message!r}')
                except BaseException as exc:
                    self.error = self.error or exc
                    if not reply.done():
                        reply.set_exception(exc)
                    raise
        finally:
            if generating is not None:
                await generating

    async def falsifier(self, rng):
        """Decide whether a draft needs scrutiny and challenge its live evidence."""
        while (request := await self.objections.get()) is not None:
            review, reply = request
            try:
                if not (review.new_plan or 'REPLAN' in review.draft.text):
                    reply.set_result(None)
                    continue
                reads = [self.common, review.recent, review.image, *review.observer.blocks,
                    *(review.plan.blocks if review.plan else []), *review.draft.blocks]
                job = await self.request('falsifier', review.step, reads)
                reply.set_result(job)
                await self.generate_tokens(job, self.params['falsifier'], rng)
            except BaseException as exc:
                self.error = self.error or exc
                if not reply.done():
                    reply.set_exception(exc)
                raise

    async def generate_tokens(self, job, params, rng):
        """Model IO only; the role chooses the job and owns its execution."""
        started = time.perf_counter()
        live = [b.name for b in job.reads if b.producer is not None and not b.producer.done]
        try:
            if self.error:
                raise self.error
            output = await self.engine.prefill(job.prompt, job.reads, job.prefix)
            for _ in range(params.budget):
                token, piece, eos = self.engine.sample(output, generator=rng,
                    temperature=params.temperature, top_k=params.top_k, top_p=params.top_p)
                job.sampled_tokens += 1
                # This chat-boundary rule belongs to this policy, not the engine.
                if eos or any(tag in piece for tag in ('<|im_end|>', '<|endoftext|>', '<|im_start|>')):
                    break
                output = await self.engine.decode(token, job.reads + [job.prefix], job.tail)
                job.tokens.append(token)
                job.text += piece
                async with job.changed:
                    job.changed.notify_all()
        except BaseException as exc:
            job.error = exc
            self.error = self.error or exc
        finally:
            try:
                self.recorder.log('stream', dict(role=job.phase, observation=job.step,
                    text=job.text, sampled_tokens=job.sampled_tokens, visible_tokens=len(job.tokens),
                    seconds=time.perf_counter()-started, live_dependencies_at_start=live,
                    error=None if job.error is None else repr(job.error)))
            except BaseException as exc:
                job.error = job.error or exc
                self.error = self.error or exc
            for block in job.reads:
                try:
                    await block.aclose()
                except BaseException as exc:
                    job.error = job.error or exc
                    self.error = self.error or exc
            job.done = True
            async with job.changed:
                job.changed.notify_all()

    async def request(self, phase, step, reads, *, last_action='none', close_previous=True):
        if self.error:
            raise self.error
        name = f'{self.context.world_seed}/{step}/{phase}'
        prefix = await BlockHandle.create(self.engine, name + '/instruction')
        try:
            tail = await BlockHandle.create(self.engine, name + '/draft')
        except BaseException:
            await prefix.aclose()
            raise
        job = Generation(phase, step, [], prefix, tail,
            role_request(phase, step, last_action, close_previous=close_previous))
        try:
            for block in reads:
                job.reads.append(block.share())
        except BaseException:
            for block in [*job.reads, prefix, tail]:
                await block.aclose()
            raise
        tail.producer = job
        self.jobs.add(job)
        return job

    async def close_generation(self, job):
        if job.closed:
            return
        # Never cancel a submitted GPU request or release blocks underneath it.
        await job.finish(raise_error=False)
        job.closed = True
        try:
            await job.prefix.aclose()
        finally:
            await job.tail.aclose()
            self.jobs.discard(job)

    async def executor(self, initial, draft_rng, refine_rng):
        """Coordinate live role answers, generate draft/refine, and act in World."""
        previous = current = initial.image
        last_action, feedback = 'none', None
        request_replan = False
        action_names = [name for name, _ in SPELEO.actions]
        action_ids = [self.engine.encode(name) for name in action_names]
        if not all(len(ids) == 1 for ids in action_ids):
            raise ValueError('Speleo action readout requires single-token action names')
        action_ids = [ids[0] for ids in action_ids]

        for step in range(self.max_actions):
            if self.stop_requested():
                break
            plan_text, plan_step, request_replan = await self.ask(
                self.plans, PlanBoundary(step, request_replan))
            owned, local = [], []
            started = time.perf_counter()
            try:
                recent = await BlockHandle.create(self.engine, f'{self.context.world_seed}/{step}/recent')
                owned.append(recent)
                image = await BlockHandle.create(self.engine, f'{self.context.world_seed}/{step}/images')
                owned.append(image)
                # Let both preparations finish even if one fails: the image buffer
                # cannot be released underneath an outstanding observer prefill.
                prepared = await asyncio.gather(
                    self.engine.prefill(recent_text(self.events, plan_text, plan_step, step),
                                        [self.common], recent),
                    self.ask(self.observations, Observe(step, previous, current, last_action, image)),
                    return_exceptions=True)
                for result in prepared:
                    if isinstance(result, BaseException):
                        raise result
                observer = prepared[1]
                local.append(observer)
                plan, new_plan = await self.ask(self.plans, PlanObservation(step, image, observer))
                if new_plan:
                    request_replan = False

                await observer.wait_tokens(4)
                reads = [self.common, recent, image, *observer.blocks, *(plan.blocks if plan else [])]
                draft = await self.request('executor_draft', step, reads)
                local.append(draft)
                await self.generate_tokens(draft, self.params['executor_draft'], draft_rng)
                await draft.finish()

                counter = await self.ask(self.objections,
                    Review(step, recent, image, observer, plan, draft, new_plan))
                if counter is not None:
                    local.append(counter)
                    await counter.wait_tokens(4)
                refined_reads = reads + draft.blocks + (counter.blocks if counter else [])
                refined = await self.request('executor_refine', step, refined_reads)
                local.append(refined)
                await self.generate_tokens(refined, self.params['executor_refine'], refine_rng)
                await refined.finish()

                readout = await BlockHandle.create(self.engine, f'{self.context.world_seed}/{step}/action')
                owned.append(readout)
                live_inputs = dict(observer=observer.text, planner=plan.text if plan else None,
                    planner_done=plan is None or plan.done, planner_based_on=plan.step if plan else None,
                    falsifier=counter.text if counter else None, falsifier_done=counter is None or counter.done)
                output = await self.engine.prefill(action_request(SPELEO, step),
                    refined_reads + refined.blocks, readout)
                scores = self.engine.score_tokens(output, action_ids)
                index = max(range(len(scores)), key=scores.__getitem__)
                action = action_names[index]
                latency = time.perf_counter() - started
                request_replan = request_replan or 'REPLAN' in draft.text or 'REPLAN' in refined.text
                self.recorder.log('decision', dict(observation=step, action_index=index, action=action,
                    action_probabilities=dict(zip(action_names, scores)), decision_seconds=latency,
                    draft=draft.text, assessment=refined.text, published_plan=plan_text,
                    published_plan_based_on=plan_step, new_planner_started=new_plan,
                    requested_replan=request_replan, live_inputs_at_action_submit=live_inputs))

                try:
                    before = time.perf_counter()
                    observation = await self.world.pass_action(index)
                    seconds = time.perf_counter()-before
                    info, height = position_info(observation.info)
                    await self.recorder.step(step=step+1, image=observation.image, action=action,
                        action_index=index, reward=observation.reward, done=observation.done,
                        info=info, height=height, decision_seconds=latency, world_seconds=seconds)
                    print('ACTION', json.dumps(dict(episode=self.context.episode_id,
                        step=step+1, height=height, action=action)), flush=True)
                finally:
                    # Record feedback for the PREVIOUS action, as in the original policy.
                    await observer.finish()
                    event = history_event(step, last_action, observer.text, feedback)
                    self.events.append(event)
                    await self.engine.prefill(event_text(event), [self.common], self.history)
                    self.recorder.log('event', event)
                last_action = action
                previous, current = current, observation.image
                feedback = dict(reward=observation.reward, episode_done=observation.done)
                if observation.done:
                    break
            finally:
                for job in local:
                    await self.close_generation(job)
                for block in owned:
                    await block.aclose()

    async def drain_roles(self):
        for inbox in (self.observations, self.plans, self.objections):
            inbox.put_nowait(None)
        failures = await asyncio.gather(*self.actors, return_exceptions=True)
        for job in list(self.jobs):
            await self.close_generation(job)
        if self.history is not None:
            await self.history.aclose()
            self.history = None
        if self.common is not None:
            await self.common.aclose()
            self.common = None
        if self.error:
            raise self.error
        for failure in failures:
            if isinstance(failure, BaseException):
                raise failure

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
            self.history = await BlockHandle.create(self.engine, f'{self.context.world_seed}/history')
            await self.engine.prefill(message('user', 'Accumulated event history begins. No actions yet.'),
                                      [self.common], self.history)
            rngs = {}
            for phase, params in self.params.items():
                seed = (1_000_000_007 * self.context.model_seed
                        + 1_000_003 * (self.context.slot+1) + params.seed_offset)
                rngs[phase] = self.engine.new_generator(seed)
                self.recorder.log('role_parameters', dict(role=phase, seed=seed,
                    temperature=params.temperature, top_k=params.top_k, top_p=params.top_p, budget=params.budget))
            self.actors = [
                asyncio.create_task(self.observer(rngs['observer']), name=f'{self.context.episode_id}/observer'),
                asyncio.create_task(self.planner(rngs['planner']), name=f'{self.context.episode_id}/planner'),
                asyncio.create_task(self.falsifier(rngs['falsifier']), name=f'{self.context.episode_id}/falsifier'),
            ]
            executor = asyncio.create_task(self.executor(initial, rngs['executor_draft'], rngs['executor_refine']),
                                           name=f'{self.context.episode_id}/executor')
            await executor
            status = 'stopped' if self.stop_requested() else 'completed'
        finally:
            try:
                await self.drain_roles()
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
