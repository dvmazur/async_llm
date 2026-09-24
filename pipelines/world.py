"""Async World facade over a native helper process (unpatched Craftium holds GIL)."""
import asyncio
from dataclasses import dataclass
import multiprocessing
import math
import os
from pathlib import Path
import subprocess
import traceback
from contextlib import contextmanager


@dataclass(frozen=True)
class Observation:
    image: object
    reward: float = 0.
    done: bool = False
    info: dict = None


@contextmanager
def _game_start_port():
    """Serialize local starts and choose a checked, non-ephemeral UDP port.

    Craftium otherwise picks an unchecked random port in the ephemeral range,
    which can collide with Luanti clients of the other live environments.
    Keep the lock until reset returns and Luanti has bound the selected port.
    """
    import fcntl
    import socket
    import tempfile
    low, high = map(int, Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    lock_path = Path(tempfile.gettempdir()) / f'craftium-game-start-{os.getuid()}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for port in range(20000, 32768):
            if low <= port <= high:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
                try:
                    reservation.bind(('0.0.0.0', port))
                except OSError:
                    continue
            yield port
            return
        raise RuntimeError('No free non-ephemeral Craftium game port')


def _make_environment(settings, root):
    if settings.get('gym_id'):
        import gymnasium
        # Registration supplies the axe, map and discrete controls. Resolution,
        # action duration and camera scale are explicit experiment overrides.
        # Editable Craftium registration assumes cwd is its build directory;
        # resolve the same task assets explicitly without changing cwd globally.
        env = gymnasium.make(settings['gym_id'], minetest_dir=str(root),
            env_dir=str(root/'craftium-envs/chop-tree'),
            max_timesteps=settings['max_steps'], offscreen_sdl=False, render_mode='rgb_array',
            **settings.get('gym_kwargs', {}))
        wrapper = env
        while not hasattr(wrapper, 'actions') and hasattr(wrapper, 'env'):
            wrapper = wrapper.env
        expected = ['forward', 'jump', 'dig', 'mouse x+', 'mouse x-', 'mouse y+', 'mouse y-']
        if env.action_space.n != 8 or getattr(wrapper, 'actions', None) != expected:
            env.close()
            raise RuntimeError('ChopTree action mapping changed; refusing incorrect policy controls')
        if 'mouse_mov' in settings:
            wrapper.mouse_mov = settings['mouse_mov']
        return env
    import craftium
    from craftium.wrappers import DiscreteActionWrapper
    return DiscreteActionWrapper(craftium.CraftiumEnv(
        env_dir=root/'craftium-envs/speleo', minetest_dir=str(root),
        obs_width=224, obs_height=224, frameskip=8, max_timesteps=settings['max_steps'],
        init_frames=200, offscreen_sdl=False, render_mode='rgb_array', _voxel_obs_available=True),
        actions=['forward', 'jump', 'mouse x+', 'mouse x-', 'mouse y+', 'mouse y-'], mouse_mov=.5*64/224)


def _serve(connection, settings):
    env = display = None
    original_popen = subprocess.Popen
    def owned_process(*args, **kwargs):
        # Craftium normally creates a new session for Luanti. Keep descendants
        # inside our GPU-worker's group so parent fail-fast can reap all of them.
        # This override exists only in this single-purpose helper process.
        kwargs['start_new_session'] = False
        return original_popen(*args, **kwargs)
    subprocess.Popen = owned_process
    try:
        import craftium
        root = Path(settings.pop('craftium_directory') or Path(craftium.__file__).resolve().parent.parent)
        if not any((root/'bin'/name).exists() for name in ('luanti', 'minetest')):
            raise FileNotFoundError(f'Craftium binary missing at {root}')
        # Match the verified portable setup; rendering must not contend with CUDA.
        os.environ.update(LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe',
            LP_NUM_THREADS='2', __GLX_VENDOR_LIBRARY_NAME='mesa')
        mesa = '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
        if Path(mesa).exists():
            os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = mesa
        if not os.environ.get('DISPLAY'):
            read_fd, write_fd = os.pipe()
            display = subprocess.Popen(['Xvfb', '-displayfd', str(write_fd), '-screen', '0', '640x480x24', '-nolisten', 'tcp'], pass_fds=(write_fd,))
            os.close(write_fd)
            with os.fdopen(read_fd) as stream:
                number = stream.readline().strip()
            if not number:
                raise RuntimeError('Xvfb failed to allocate a display')
            os.environ['DISPLAY'] = ':' + number
        env = _make_environment(settings, root)
        env.unwrapped.mt.proc_env = dict(os.environ, SDL_VIDEODRIVER='x11')
        env.unwrapped.mt.overwrite_config({'video_driver': 'opengl3'})
        import hashlib
        import mt_server
        revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        connection.send(('ok', dict(craftium_root=str(root), craftium_revision=revision,
            environment=settings.get('gym_id', 'Craftium/Speleo-v0'),
            observation_shape=list(env.observation_space.shape), action_count=int(env.action_space.n),
            environment_overrides=settings.get('gym_kwargs', {}),
            mouse_mov=settings.get('mouse_mov', .5*64/224),
            mt_server_sha256=hashlib.sha256(Path(mt_server.__file__).read_bytes()).hexdigest())))
        while True:
            operation, value = connection.recv()
            if operation == 'close':
                break
            if operation == 'reset':
                if settings.get('gym_id'):
                    with _game_start_port() as port:
                        env.unwrapped.mt.overwrite_config({'port': port, 'remote_port': port})
                        image, info = env.reset(seed=value)
                else:
                    image, info = env.reset(seed=value)
                result = Observation(image.copy(), info=info)
            elif operation == 'step':
                image, reward, terminated, truncated, info = env.step(value)
                info = dict(info, terminated=bool(terminated), truncated=bool(truncated))
                result = Observation(image.copy(), float(reward), bool(terminated or truncated), info)
            else:
                raise ValueError(operation)
            connection.send(('ok', result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        try:
            if env:
                env.close()
        finally:
            if display:
                display.terminate()
                display.wait(timeout=5)
            connection.close()
            subprocess.Popen = original_popen


class SpeleoWorld:
    def __init__(self, *, seed, max_steps=100, craftium_directory=None):
        self.seed = seed
        self.settings = dict(max_steps=max_steps, craftium_directory=craftium_directory)
        self.process = self.connection = None
        self.metadata = {}
        self.busy = self.closed = False

    async def _receive(self):
        kind, value = await asyncio.to_thread(self.connection.recv)
        if kind != 'ok':
            raise RuntimeError(value)
        return value

    async def _call(self, operation, value):
        if self.closed or self.busy:
            raise RuntimeError('World is closed or already has an unfinished operation')
        self.busy = True
        try:
            if self.process is None:
                ctx = multiprocessing.get_context('spawn')
                self.connection, child = ctx.Pipe()
                self.process = ctx.Process(target=_serve, args=(child, dict(self.settings)))
                self.process.start()
                child.close()
                self.metadata = await self._receive()
            self.connection.send((operation, value))
            return await self._receive()
        finally:
            self.busy = False

    async def reset(self):
        return await self._call('reset', self.seed)

    async def pass_action(self, action):
        return await self._call('step', action)

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        if self.process:
            try:
                if self.process.is_alive():
                    self.connection.send(('close', None))
                await asyncio.to_thread(self.process.join, 10)
                if self.process.is_alive():
                    self.process.terminate()
                    await asyncio.to_thread(self.process.join, 5)
                if self.process.is_alive():
                    self.process.kill()
                    await asyncio.to_thread(self.process.join)
            finally:
                self.connection.close()


class ChopTreeWorld(SpeleoWorld):
    """Same isolated native IPC transport, registered ChopTree instead of Speleo."""
    def __init__(self, *, seed, max_steps=2000, craftium_directory=None, frameskip=8,
                 pmul=20, turn_degrees=10):
        if type(frameskip) is not int or frameskip < 1:
            raise ValueError('frameskip must be a positive integer')
        if not math.isfinite(pmul) or pmul <= 0:
            raise ValueError('pmul must be finite and positive')
        if not math.isfinite(turn_degrees) or turn_degrees <= 0:
            raise ValueError('turn_degrees must be finite and positive')
        super().__init__(seed=seed, max_steps=max_steps, craftium_directory=craftium_directory)
        self.settings['gym_id'] = 'Craftium/ChopTree-v0'
        # Keep the requested turn angle independent of movement/dig duration.
        # Keeping the integer mouse displacement avoids rounding to 9 or 11 deg.
        # ChopTree's on-join hook sets noon (0.5). Freeze that clock so API/model
        # latency cannot consume daylight. This does not change physics timing.
        self.settings['gym_kwargs'] = dict(obs_width=224, obs_height=224, frameskip=frameskip,
            pmul=pmul, minetest_conf={'mouse_sensitivity': .2*turn_degrees/(4.48*frameskip),
                                  'time_speed': 0})
        self.settings['mouse_mov'] = .5*64/224
