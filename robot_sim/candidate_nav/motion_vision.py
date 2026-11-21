"""Read-only motion image sampling, independent of ROS and motor commands."""
import math
from collections import deque
from .prepared_perception import image_digest
from . import target_vision

PROTOCOL = 'motion_visual_memory_v1'

class MotionFrames:
    def __init__(self):
        self.pending = deque(maxlen=128)
        self.last = None
        self.serial = 0
        self.dropped = 0

    def add(self, image, stamp, pose, plan_id, force=False):
        if self.last is not None:
            old_stamp, old_pose = self.last
            if stamp <= old_stamp:
                return None
            angle = abs(math.atan2(math.sin(pose[2]-old_pose[2]), math.cos(pose[2]-old_pose[2])))
            if not force and angle < math.pi/16 and math.dist(pose[:2],old_pose[:2]) < .10:
                return None
        self.serial += 1
        meta = dict(frame_id=self.serial, stamp=stamp, pose=list(pose), plan_id=plan_id)
        if len(self.pending) == self.pending.maxlen:
            self.dropped += 1
        self.pending.append((meta, image.copy() if hasattr(image, "copy") else image))
        self.last = (stamp, list(pose))
        return meta

    def take(self, count=4):
        return [self.pending.popleft() for _ in range(min(count,len(self.pending)))]


import asyncio
import base64
import io

def image64(image):
    buffer=io.BytesIO()
    image.save(buffer,format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

class MotionObserver:
    def prepare_motion_frames(self, batch):
        output = getattr(self, 'output', None)
        if output is not None:
            for meta, image in batch:
                image.save(output / 'motion_frames' / f"{meta['frame_id']:06d}.jpg", quality=90)
        return [dict(meta, image=image64(image), image_sha256=image_digest(image)) for meta,image in batch]

    async def next_motion_batch(self):
        while not self.finished.is_set():
            batch=self.motion_frames.take()
            if batch:
                self.motion_preparing=True
                try:return batch,await asyncio.to_thread(self.prepare_motion_frames,batch)
                finally:self.motion_preparing=False
            self.vision_wakeup.clear()
            if not getattr(self,'motion_inflight',False):self.vision_idle.set()
            await self.vision_wakeup.wait()
        return [],[]

    async def motion_observer(self):
        prefetch=asyncio.create_task(self.next_motion_batch())
        try:
            await self._observe_batches(prefetch)
        finally:
            task=getattr(self,'motion_prefetch',prefetch)
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)

    async def _observe_batches(self,prefetch):
        self.motion_prefetch=prefetch
        while not self.finished.is_set():
            batch,prepared=await prefetch
            if not batch:return
            # Frames arriving while the GPU worked must still share a batch.
            # Prefetching one early image must not turn a backlog into one RPC
            # per frame. Keep chronology and the original four-frame cap.
            extra=self.motion_frames.take(4-len(batch))
            if extra:
                prepared.extend(await asyncio.to_thread(self.prepare_motion_frames,extra))
                batch.extend(extra)
            self.vision_idle.clear()
            self.motion_inflight=True
            prefetch=asyncio.create_task(self.next_motion_batch())
            self.motion_prefetch=prefetch
            while getattr(self,'plan_request_pending',False) and not self.finished.is_set():
                await asyncio.sleep(.01)
            try:
                status = getattr(self, 'perception_status', None)
                if status is not None:
                    status.update(state='processing', pending_frames=len(self.motion_frames.pending),
                                  in_flight_frames=len(batch))
                packet = dict(protocol=PROTOCOL, snapshot_id=batch[-1][0]['frame_id'],
                              map_version=self.version, instruction=self.task['instruction'],
                              frames=prepared)
                fields=getattr(self,'memory_packet_fields',None)
                if fields is not None:packet.update(fields())
                previous = getattr(self, 'motion_previous_frame', None)
                if previous is not None:
                    packet['previous_frame'] = self.motion_previous_encoded
                packet['visual_state'] = getattr(self, 'motion_visual_state', {})
                if packet.get('physical') is True:
                    packet['visual_tracking'] = target_vision.MODE
                    packet['target_reference'] = getattr(self, 'target_reference', None)
                    packet.pop('previous_frame', None)
                reply = await self.request_motion_observation(packet)
                memory = dict(evidence=reply['evidence'], frames=[{k:v for k,v in meta.items() if k!='execution'} for meta,_ in batch],
                              source='historical_motion_camera', current_visibility=False)
                self.motion_previous_frame = batch[-1]
                self.motion_previous_encoded = prepared[-1]
                if isinstance(reply.get('visual_state'), dict):
                    self.motion_visual_state = reply['visual_state']
                    if target_vision.is_split_state(reply['visual_state']):
                        self.target_reference = target_vision.remember_reference(
                            getattr(self, 'target_reference', None), reply['visual_state'],
                            {meta['frame_id']: image for meta, image in batch})
                    else:
                        memory['visual_state'] = reply['visual_state']
                if reply.get('events'):
                    self.visual_revision = getattr(self,'visual_revision',0)+1
                    stamps=[e.get('frame',{}).get('stamp',0) for e in reply['events']]
                    self.visual_revision_stamp=max([getattr(self,'visual_revision_stamp',0),*stamps])
                if reply.get('events') or ('events' not in reply and reply['evidence'] != 'nothing new'):
                    self.visual_memory.append(memory)
                self.motion_last_result = (*batch[-1], reply, packet['frames'][-1]['image_sha256'])
                self.motion_processed_frame_id = batch[-1][0]['frame_id']
                self.motion_processed_count = getattr(self, 'motion_processed_count', 0)+len(batch)
                if status is not None:
                    status.update(state='ready', processed_frames=self.motion_processed_count,
                                  pending_frames=len(self.motion_frames.pending), in_flight_frames=0,
                                  last_frame_id=self.motion_processed_frame_id,
                                  last_evidence=reply['evidence'])
                self.log('events',dict(event='motion_visual_evidence',**memory,latency_s=reply.get('latency_s'),
                                      during_motion=getattr(self,'trajectory_progress',None) is not None,
                                      processed_frames=self.motion_processed_count))
            except Exception as error:
                self.motion_failed_frame_id = batch[-1][0]['frame_id']
                self.log('events',dict(event='motion_vision_error',error=repr(error),frames=[m for m,_ in batch]))
            finally:
                self.motion_inflight=False
                next_ready=(prefetch.done() and not prefetch.cancelled() and prefetch.exception() is None and bool(prefetch.result()[0]))
                if not self.motion_frames.pending and not getattr(self,'motion_preparing',False) and not next_ready:self.vision_idle.set()
                progress = getattr(self, 'vision_progress', None)
                if progress is not None:
                    progress.set()
            # These results never enqueue a driving command or cancel a sequence.

    async def flush_motion_vision(self):
        self.state = 'finishing_motion_observations'
        try:
            await asyncio.wait_for(self.vision_idle.wait(),120)
        except asyncio.TimeoutError:
            self.log('events',dict(event='motion_vision_flush_timeout'))
