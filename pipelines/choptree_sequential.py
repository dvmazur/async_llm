"""One Qwen conversation: shared within a step, optionally retained across steps."""
from experiment_runner.blocks import BlockHandle
from experiment_runner.probe_readout import ProbeReadout
from .choptree import ACTION_NAMES
from .choptree_target_actor import TargetActorReadout, TARGET_PROMPT, MOTOR_PROMPT, marked_view


CHAT_END = '<|im_end|>\n'
SYSTEM_PROMPT = ('You control one ChopTree agent through three stages in ONE conversation. '
    'At PLANNER, select a target; at ACTOR, assess the next operation; at ACTION, '
    'return only one legal control. Old observations are history, not the current view.\n\n'
    'PLANNER instructions:\n'+TARGET_PROMPT+'\n\nACTOR instructions:\n'+MOTOR_PROMPT+
    '\n\nACTION instructions: Choose the ONE immediate action using the latest plan '
    'and assessment. An incomplete assessment is not an instruction to wait. '
    'Return only one of: '+', '.join(ACTION_NAMES)+'.')


class ContextLimitError(RuntimeError):
    pass


class CheckedChatReadout(ProbeReadout):
    def __init__(self, backend, reserve):
        super().__init__(backend)
        self.reserve = reserve

    async def prefill_action(self, messages, target):
        inputs = self.prepare_inputs(messages)
        rows = inputs['input_ids'].numel()
        limit = self.backend.llm.engine.config.max_seq_len
        required = target.raw.num_tokens + rows + self.reserve
        if required > limit:
            raise ContextLimitError(f'Sequential context needs {required} tokens including '
                f'generation reserve; model limit is {limit}. History was NOT reset or trimmed.')
        output = await self.backend.llm(**inputs, cache_view=[target.raw])
        return output, rows


class SequentialTargetActorReadout(TargetActorReadout):
    def __init__(self, backend, recorder, memory, *, history_scope, **kwargs):
        if history_scope not in ('step','episode'):
            raise ValueError('history_scope must be step or episode')
        super().__init__(backend, recorder, memory, **kwargs)
        self.history_scope = history_scope
        self.has_history = history_scope == 'episode'
        self.chat = None
        self.close_rows = len(self.encode(CHAT_END))
        self.planner.readout = CheckedChatReadout(backend, self.max_role_tokens+self.close_rows)
        self.actor.readout = CheckedChatReadout(backend, self.max_role_tokens+self.close_rows)
        self.action_readout = CheckedChatReadout(backend, 1+self.close_rows)
        self.policy_metadata.update(mode='planner-actor-single-chat-'+history_scope,
            history_scope=history_scope, automatic_trimming=False,
            reset_between_actions=history_scope=='step', automatic_reset_on_overflow=False,
            vision='current full448 frame appended at PLANNER; ACTOR sees the same chat')

    @property
    def history_tokens(self):
        return self.chat.raw.num_tokens if self.chat is not None else 0

    async def close_episode(self):
        if self.chat is not None:
            await self.chat.aclose()
            self.chat = None

    async def _end_turn(self):
        # TextRole.on_block retained the last content token, even at the budget.
        # EOS/control boundary tokens are not decoded by that helper: close exactly once.
        await self.backend.prefill(CHAT_END, [], self.chat)

    async def generate_action(self, messages, target, *, generator, temperature):
        if self.history_scope == 'step':
            # Reuse the caller-owned empty action block; never keep it after this call.
            self.chat = target
        elif self.chat is None:
            self.chat = await BlockHandle.create(self.backend,'sequential/conversation')
        try:
            first = self.chat.raw.num_tokens == 0
            before = self.history_tokens
            frames=[p['image'] for p in messages[1]['content'] if p['type']=='image']
            self.memory.inspect_change(frames[0],frames[-1])
            visual=self.memory.visual_outcome
            recent=self.memory.recent[-1] if self.memory.recent else dict(action='none',reward=0)
            if self.history_scope == 'step':
                # Match the fresh-context policy's bounded external memory.
                feedback=self.memory.text()
            else:
                # Previous plans, actions, rewards and images are already in the chat.
                feedback=(f'Last action={recent["action"]}, reward={recent["reward"]:g}; '
                    f'wood={self.memory.total_reward:g}, since_wood={self.memory.since_reward}. '
                    f'RGB change={visual["normalized_rgb_change"]}, '
                    f'nearly unchanged={visual["almost_unchanged"]}, '
                    f'revisited views (actions ago)={visual["similar_view_actions_ago"]}.')
            planner_messages=([dict(role='system',content=SYSTEM_PROMPT)] if first else [])+[
                dict(role='user',content=[
                    dict(type='text',text=f'PLANNER. Current observation {self.memory.step}. Red cross is aim.'),
                    dict(type='image',image=marked_view(frames[-1])),
                    dict(type='text',text='Public feedback: '+feedback)])]
            plan,rows1=await self.planner.on_block(planner_messages,self.chat,
                generator=generator,max_tokens=self.max_role_tokens,
                temperature=self.recovery_temperature if self.memory.since_reward>=self.recovery_after
                else self.planner_temperature)
            await self._end_turn()
            assessment,rows2=await self.actor.on_block([
                dict(role='user',content='ACTOR. Assess the latest target and immediate operation.')],
                self.chat,generator=generator,temperature=temperature,max_tokens=self.max_role_tokens)
            await self._end_turn()
            output,rows3=await self.action_readout.prefill_action([
                dict(role='user',content='ACTION. Choose only the next control.')],self.chat)
            if self.action_ids is None:
                ids=[self.encode(name) for name in ACTION_NAMES]
                if not all(len(x)==1 for x in ids) or len({x[0] for x in ids})!=len(ids):
                    raise ValueError('actions must be distinct single tokens')
                self.action_ids=[x[0] for x in ids]
            index,probs=self.sample_action(output,self.action_ids,
                generator=generator,temperature=temperature)
            # Sampling does not append the selected action to KV.
            await self.backend.decode(self.action_ids[index],[],self.chat)
            await self._end_turn()
            action=ACTION_NAMES[index]
            self.memory.last_plan=dict(plan=plan,assessment=assessment,action=action)
            self.recorder.log('target_decision',dict(step=self.memory.step,plan=plan,
                assessment=assessment,decision=dict(action=action),history_scope=self.history_scope,
                context_tokens_before=before,context_tokens_after=self.history_tokens,
                action_probabilities=dict(zip(ACTION_NAMES,probs))))
            return index,rows1+rows2+rows3+3*self.close_rows
        finally:
            if self.history_scope == 'step':
                self.chat = None  # outer pipeline owns this block and frees it
