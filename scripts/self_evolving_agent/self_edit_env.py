from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from minisgl.llm import AsyncLLM

logger = logging.getLogger(__name__)

# Names on the Engine class we never copy over during a live reload: __init__
# is deliberately never re-run against a live instance (that would defeat the
# whole point -- state like cache blocks must survive), and the rest are
# Python bookkeeping attributes every class carries that have nothing to do
# with the agent's own behavior.
_RELOAD_SKIP_NAMES = {"__init__", "__module__", "__qualname__", "__dict__", "__weakref__", "__doc__"}


class SelfEditEnv:
    """Holds one live, persistent Engine instance for the whole process.
    Unlike a plain module reload, `reload_engine_methods` patches new method
    bodies onto the instance's own class in place -- the instance itself
    (and everything it holds: cache blocks, memory, counters, ...) is never
    torn down or reconstructed after the first successful load."""

    def __init__(
        self,
        llm: "AsyncLLM",
        engine_path: str = "mutable/engine.py",
        prompt_path: str = "mutable/prompt.py",
    ) -> None:
        self.llm = llm
        self.engine_path = Path(engine_path)
        self.prompt_path = Path(prompt_path)
        self.screen: Optional[str] = None
        self.prompt_source: Optional[str] = None
        self.prompt: Optional[Any] = None
        self.engine: Optional[Any] = None
        self._engine_cls: Optional[type] = None
        self.load_error: Optional[str] = None
        self.prompt_load_error: Optional[str] = None

        # Not agent-editable -- set externally (e.g. by run_persistent.py),
        # exactly like engine_path/prompt_path. task_env is a doom_basic-style
        # GymEnv (see tasks/math_env.py); on_episode_start is an optional
        # duck-typed heartbeat hook (same pattern as TTFTEnv.on_token) so a
        # caller's hang-watchdog keeps getting touched during a long
        # start_task()/restart_task() call.
        self.task_env: Optional[Any] = None
        self.on_episode_start: Optional[Callable[[], None]] = None
        self.on_step: Optional[Callable[[], None]] = None
        self.last_task_result: Optional[dict[str, Any]] = None

    def _compile_engine_class(self, source: str) -> type:
        """Exec source into a throwaway namespace (never touching
        sys.modules) and return its Engine class, without instantiating it."""
        namespace: dict[str, Any] = {"__name__": "engine"}
        code = compile(source, str(self.engine_path), "exec")
        exec(code, namespace)
        return namespace["Engine"]

    def _compile_prompt_class(self, source: str) -> Any:
        """Exec source into a throwaway namespace, extract its top-level
        Prompting class, and construct Prompting(self.llm). Unlike
        _compile_engine_class this also constructs the instance -- Prompting
        holds no persistent state worth preserving across reloads (see
        reload_prompt), so there is no reason to hand back the bare class."""
        namespace: dict[str, Any] = {"__name__": "prompt"}
        code = compile(source, str(self.prompt_path), "exec")
        exec(code, namespace)
        cls = namespace["Prompting"]
        instance = cls(self.llm)
        text = getattr(instance, "system_prompt", None)
        if not isinstance(text, str):
            raise TypeError(f"Prompting.system_prompt must be a str, got {type(text).__name__}")
        # A real prompt is thousands of chars and documents the <use_tool>
        # syntax -- reject anything that looks like a stub/placeholder
        # instead of silently accepting it and leaving the agent unable to
        # call any tool on the next round.
        if len(text) < 500:
            raise ValueError(f"Prompting.system_prompt looks broken: only {len(text)} chars "
                              f"(expected a real prompt)")
        if "<use_tool" not in text:
            raise ValueError("Prompting.system_prompt is missing the '<use_tool' tool-call "
                              "documentation -- reloading this would leave you unable to call any tool")
        return instance

    def reset(self) -> str:
        self.screen = self.engine_path.read_text()
        try:
            cls = self._compile_engine_class(self.screen)
            self.engine = cls(self.llm)
            self._engine_cls = cls
            self.load_error = None
            logger.info("loaded initial %s (%d chars)", self.engine_path, len(self.screen))
        except Exception:
            self.engine = None
            self._engine_cls = None
            self.load_error = traceback.format_exc()
            logger.error("initial %s does not load:\n%s", self.engine_path, self.load_error)

        self.prompt_source = self.prompt_path.read_text()
        try:
            self.prompt = self._compile_prompt_class(self.prompt_source)
            self.prompt_load_error = None
            logger.info("loaded initial %s (%d chars)", self.prompt_path, len(self.prompt_source))
        except Exception:
            self.prompt = None
            self.prompt_load_error = traceback.format_exc()
            logger.error("initial %s does not load:\n%s", self.prompt_path, self.prompt_load_error)
        return self.screen

    def write_engine(self, source: str) -> dict[str, Any]:
        """Tool 1: persist a new engine.py source. Does not take effect until
        reload_engine_methods() is called."""
        self.engine_path.write_text(source)
        self.screen = source
        logger.info("wrote %s (%d chars)", self.engine_path, len(source))
        return {"ok": True}

    def write_prompt(self, source: str) -> dict[str, Any]:
        """Tool 3: persist a new prompt.py source. Does not take effect until
        reload_prompt() is called -- same two-step shape as
        write_engine/reload_engine_methods."""
        self.prompt_path.write_text(source)
        self.prompt_source = source
        logger.info("wrote %s (%d chars)", self.prompt_path, len(source))
        return {"ok": True}

    def reload_prompt(self) -> dict[str, Any]:
        """Tool 4: recompile the on-disk prompt.py, extract its Prompting
        class, and construct a fresh Prompting(llm) instance for the next
        round. There is no live instance state to preserve here (unlike
        reload_engine_methods) -- Prompting holds nothing but strings derived
        from llm, so reload just rebuilds it from scratch each time rather
        than patching methods onto an old instance."""
        source = self.prompt_path.read_text()
        try:
            new_prompt = self._compile_prompt_class(source)
        except Exception:
            tb = traceback.format_exc()
            logger.error("failed to compile %s:\n%s", self.prompt_path, tb)
            return {"ok": False, "error": tb}
        self.prompt = new_prompt
        self.prompt_load_error = None
        logger.info("reloaded %s (%d chars)", self.prompt_path, len(source))
        return {"ok": True}

    def reload_engine_methods(self) -> dict[str, Any]:
        """Tool 2: recompile the on-disk engine.py and patch its methods onto
        the already-live Engine instance in place. __init__ is never re-run
        once an instance exists -- only method bodies are swapped, so
        instance state (cache blocks, memory, ...) survives untouched."""
        source = self.engine_path.read_text()
        try:
            new_cls = self._compile_engine_class(source)
        except Exception:
            tb = traceback.format_exc()
            logger.error("failed to compile %s:\n%s", self.engine_path, tb)
            return {"ok": False, "error": tb}

        if self.engine is None:
            try:
                self.engine = new_cls(self.llm)
            except Exception:
                tb = traceback.format_exc()
                logger.error("failed to construct Engine from %s:\n%s", self.engine_path, tb)
                return {"ok": False, "error": tb}
            self._engine_cls = new_cls
            self.load_error = None
            logger.info("constructed initial live Engine instance")
            return {"ok": True, "changed": ["__init__"]}

        changed = []
        for name, value in vars(new_cls).items():
            if name in _RELOAD_SKIP_NAMES:
                continue
            setattr(self._engine_cls, name, value)
            changed.append(name)
        self.load_error = None
        logger.info("patched methods on live Engine instance: %s", changed)
        return {"ok": True, "changed": changed}

    async def start_task(self) -> dict[str, Any]:
        """Tool 5: run the currently-live Engine's act() against task_env
        end-to-end (all episodes) and report the score. No guardrails --
        callable any time, in any state, just like every other tool here;
        a bad/no-op engine just scores poorly rather than being blocked."""
        if self.task_env is None:
            return {"ok": False, "error": "no task_env configured"}
        from tasks.runner import run_episodes
        result = await run_episodes(self.task_env, self.engine,
                                     on_episode_start=self.on_episode_start, on_step=self.on_step)
        self.last_task_result = result
        logger.info("start_task on %s: avg_reward=%.3f", result["env"], result["avg_reward"])
        return {"ok": True, "result": result}

    async def restart_task(self) -> dict[str, Any]:
        """Tool 6: like start_task, but first resets task_env's episode
        cursor back to the beginning (see MathEnv.restart()) so the re-run
        is a clean, comparable full pass instead of resuming mid-cycle --
        the natural thing to call after adjusting Engine in response to a
        start_task score."""
        if self.task_env is None:
            return {"ok": False, "error": "no task_env configured"}
        restart = getattr(self.task_env, "restart", None)
        if callable(restart):
            restart()
        return await self.start_task()

    async def end_task(self) -> dict[str, Any]:
        """Tool 7: no-op marker the agent can call to say "done evaluating
        for this round" -- returns the last score for reference. Nothing
        else keys off having been called; there is no hidden state machine
        gating start_task/restart_task on this."""
        return {"ok": True, "result": self.last_task_result}
