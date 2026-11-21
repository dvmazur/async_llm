"""ROS-independent live-model wire protocol; shared by rover and CPU diagnostics."""

import asyncio
import json
from .task_policy import is_task_mode, MODE
from .mission_context import PROTOCOLS as MISSION_PROTOCOLS, allowed_reply


class ModelResponseError(RuntimeError):
    def __init__(self, response):
        super().__init__(response['error'])
        self.code = response.get('error_code')
        self.retryable = response.get('retryable', True)
        self.details = response


def validate_model_response(response, observation):
    if response.get("error"):
        raise ModelResponseError(response)
    if response.get("snapshot_id") != observation["snapshot_id"]:
        raise ValueError("Mismatched snapshot")
    if response.get("map_version") != observation["map_version"]:
        raise ValueError("Mismatched map version")
    if observation.get("protocol") in MISSION_PROTOCOLS:
        if not allowed_reply(observation, response):
            raise ValueError("Invalid mission worker selection")
        return response
    if observation.get("protocol") == "motion_visual_memory_v1":
        if response.get("protocol") != "motion_visual_memory_v1" or not isinstance(
            response.get("evidence"), str
        ):
            raise ValueError("Invalid motion visual observation")
        if any(k in response for k in ("actions", "action", "goal_id")):
            raise ValueError("Visual observation must not contain motor decisions")
        return response
    if observation.get("protocol") == "action_trajectory_v1":
        from .action_plan import validate_response

        validate_response(response, observation)
        return response
    if observation.get("protocol") == "direct_navigation_v1":
        from .direct_control import validate_response

        validate_response(response, observation)
        return response
    if observation.get("protocol") == "candidate_generation_v1":
        if not isinstance(response.get("proposals"), list) or len(response["proposals"]) > 8:
            raise ValueError("Invalid generated proposal list")
        return response
    if response.get("goal_id") not in [c["id"] for c in observation["candidates"]] + ["W", "Z"]:
        raise ValueError("Goal outside candidate allowlist")
    if response["goal_id"] == "Z":
        if is_task_mode(observation):
            if (
                response.get("completion_mode") != MODE
                or response.get("task_complete") is not True
            ):
                raise ValueError("Task completion response is inconsistent")
        elif not any(o.get("visible") and o["range_m"] < 0.95 for o in observation["objects"]):
            raise ValueError("Completion lacks nearby measured RGB object")
    return response


import time
import weakref
from .wire_images import WireImages
_POOLS=weakref.WeakKeyDictionary()
_PENDING=weakref.WeakKeyDictionary()

async def _close(writer):
    writer.close()
    try:await writer.wait_closed()
    except (OSError,ConnectionError):pass

async def request_model(observation, port, timeout=300, socket_path=None, on_update=None):
    """Reuse confirmed connections; a streamed plan is returned before its comment.

    A connection stays leased until its final response is consumed. A second
    request never sees another plan's late message. No request is retried after
    writing, since delivery of an action might already have succeeded.
    """
    loop=asyncio.get_running_loop();pool=_POOLS.setdefault(loop,{})
    persistent=observation.get('physical') is True
    key=(port,socket_path);idle=pool.setdefault(key,[]) if persistent else [];pair=None
    while idle:
        candidate=idle.pop()
        if not candidate[0].at_eof() and not candidate[1].is_closing():
            pair=candidate;break
        await _close(candidate[1])
    started=time.monotonic();reused=pair is not None
    if pair is None:
        connection=(asyncio.open_connection('127.0.0.1',port,limit=5_000_000)
                    if socket_path is None else asyncio.open_unix_connection(socket_path,limit=5_000_000))
        reader,writer=await asyncio.wait_for(connection,10)
        pair=(reader,writer,WireImages())
    reader,writer,image_cache=pair;leased=True
    async def read():
        data=await asyncio.wait_for(reader.readline(),timeout)
        if not data:raise RuntimeError('Model connection closed')
        response=json.loads(data);validate_model_response(response,observation)
        return response
    async def release(response):
        if persistent and response.get('transport_keepalive') and len(idle)<2 and not writer.is_closing():idle.append(pair)
        else:await _close(writer)
    try:
        wire=image_cache.encode({**observation,'_transport_v':2 if persistent else 1},enabled=reused)
        writer.write((json.dumps(wire,allow_nan=False,separators=(',',':'))+'\n').encode())
        await writer.drain();sent=time.monotonic();response=await read()
        response['transport_timing']=dict(connection_reused=reused,send_s=sent-started,round_trip_s=time.monotonic()-started)
        if response.get('message_kind')=='plan_ready':
            async def finish():
                try:
                    final=await read()
                    if final.get('message_kind')!='rationale_ready' or final.get('actions')!=response.get('actions'):
                        raise ValueError('Late response changed the committed plan')
                    if on_update is not None:on_update(final)
                    await release(final)
                except asyncio.CancelledError:
                    await _close(writer)
                except Exception as error:
                    await _close(writer)
                    if on_update is not None:
                        failed={**response,'message_kind':'rationale_error','rationale_pending':False,
                            'reasoner':'Пояснение недоступно. План уже передан.',
                            'spatial_rationale':'Пояснение недоступно. План уже передан.',
                            'explanation_error':str(error)}
                        try:on_update(failed)
                        except Exception:pass
                    # A late comment cannot revoke or duplicate a delivered plan.
            task=asyncio.create_task(finish(),name='model_rationale_update')
            pending=_PENDING.setdefault(loop,set());pending.add(task);task.add_done_callback(pending.discard)
        else:await release(response)
        leased=False
        return response
    finally:
        if leased:await _close(writer)

async def close_model_connections():
    loop=asyncio.get_running_loop()
    tasks=list(_PENDING.pop(loop,set()))
    for task in tasks:task.cancel()
    await asyncio.gather(*tasks,return_exceptions=True)
    for idle in _POOLS.pop(loop,{}).values():
        for _,writer,_ in idle:await _close(writer)
