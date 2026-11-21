"""Preflight before a physical mission. Never publishes velocity commands."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from .startup_health import LIMITS,LABELS,blockers,odom_valid,preflight,lock_run,restart_mapping


def main():
    # All normal run flags can pass through; only startup's inputs are consumed.
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rover-id',type=int,default=80)
    parser.add_argument('--instruction',default='')
    parser.add_argument('--lock-fd',type=int,required=True)
    args,_=parser.parse_known_args()
    lock_run(args.lock_fd)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
    from rclpy.time import Time
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry,OccupancyGrid
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer,TransformListener
    from slam_toolbox.srv import SerializePoseGraph
    from .live import Camera
    evidence=args.output.with_name(args.output.name+'-startup')
    evidence.mkdir(parents=True,exist_ok=True)
    rclpy.init()

    class Probe(Node):
        def __init__(self):
            super().__init__('rover_sensor_startup')
            self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
            self.camera=Camera();self.latest={};self.invalid_streak=0;self.map_message=None
            self.motion_at=-math.inf;self.lidar_frame=None;self.rays=0
            self.mapping_after_stamp=-math.inf
            self.telemetry=self.create_publisher(String,'/rover_live/telemetry',1)
            self.records=[]
            latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
            for topic,kind,name,qos in [('/scan',LaserScan,'scan',qos_profile_sensor_data),
                    ('/icp/odom',Odometry,'odom',qos_profile_sensor_data),('/map',OccupancyGrid,'map',latched)]:
                self.create_subscription(kind,topic,lambda m,n=name:self.receive(n,m),qos)
            for topic in ('/cmd_vel','/cmd_vel_requested'):
                self.create_subscription(Twist,topic,self.velocity,qos_profile_sensor_data)
        def receive(self,name,msg):
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
            if name in ('map','odom') and stamp<=self.mapping_after_stamp:
                return
            self.latest[name]=(msg,time.monotonic())
            if name=='map':self.map_message=msg
            if name=='odom':self.invalid_streak=0 if odom_valid(msg) else self.invalid_streak+1
            if name=='scan':
                self.lidar_frame=msg.header.frame_id
                self.rays=sum(math.isfinite(x) and msg.range_min<=x<=msg.range_max for x in msg.ranges)
        def velocity(self,msg):
            if any(abs(v)>1e-4 for v in (msg.linear.x,msg.linear.y,msg.linear.z,msg.angular.x,msg.angular.y,msg.angular.z)):
                self.motion_at=time.monotonic()
        def poll(self,seconds):rclpy.spin_once(self,timeout_sec=seconds)
        def snapshot(self):
            wall=time.monotonic();now=self.get_clock().now().nanoseconds/1e9
            ages={}
            for name,(msg,received) in self.latest.items():
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
                # Both transport liveness and the actual measurement stamp matter.
                stamp_age=now-stamp
                ages[name]=stamp_age if stamp_age<-.15 else max(stamp_age,wall-received)
            frame=self.camera.latest
            ages['camera']=wall-frame[2] if frame is not None else None
            lidar_tf=False
            try:
                if self.lidar_frame:self.buffer.lookup_transform('base_link',self.lidar_frame,Time());lidar_tf=bool(self.lidar_frame)
            except Exception:pass
            try:
                tf=self.buffer.lookup_transform('map','base_link',Time())
                ages['pose']=now-(tf.header.stamp.sec+tf.header.stamp.nanosec/1e9)
            except Exception:ages['pose']=None
            return dict(ages={k:round(v,4) if v is not None else None for k,v in ages.items()},
                        lidar_tf=lidar_tf,odom_valid='odom' in self.latest and odom_valid(self.latest['odom'][0]),
                        invalid_odom_streak=self.invalid_streak,valid_scan_rays=self.rays,
                        external_motion=wall-self.motion_at<3)
        def status(self,state,snapshot,attempted=False):
            missing=blockers(snapshot)
            report=dict(state=state,stamp=time.time(),blocked=missing,
                        blocked_labels=[LABELS[x] for x in missing],sensors=snapshot,recovery_attempted=attempted)
            self.records.append(report)
            (evidence/'status.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
            with (evidence/'events.jsonl').open('a') as f:f.write(json.dumps(report,ensure_ascii=False)+'\n')
            print('Sensors: '+state+(' · '+', '.join(report['blocked_labels']) if missing else ''),flush=True)
            self.telemetry.publish(String(data=json.dumps(dict(report,run=args.output.name,rover_id=args.rover_id,
                finished=state=='sensor_startup_failed',reason='sensor_startup_failed' if state=='sensor_startup_failed' else None,
                motion_enabled=False,instruction=args.instruction,sensor_startup=report),ensure_ascii=False)))
        def recover(self):
            # Capture configuration/logs and the graph before replacing only mapping.
            for name in ('icp.log','slam.log'):
                path=Path('/opt/rover-mapping-quality')/name
                if path.exists():
                    with path.open('rb') as src:
                        src.seek(max(0,path.stat().st_size-262144))
                        (evidence/name).write_bytes(src.read())
            metadata=dict(map_observed=self.map_message is not None,reason='invalid_icp_before_mission')
            client=self.create_client(SerializePoseGraph,'/slam_toolbox/serialize_map')
            try:
                if not client.wait_for_service(timeout_sec=3):raise RuntimeError('SLAM archive service unavailable')
                req=SerializePoseGraph.Request();req.filename=str(evidence/'slam-before')
                future=client.call_async(req);rclpy.spin_until_future_complete(self,future,timeout_sec=6)
                if not future.done() or future.exception() or future.result().result!=0:
                    raise RuntimeError('SLAM graph archive failed')
                metadata['graph_saved']=True
            except RuntimeError as error:
                metadata.update(graph_saved=False,archive_error=str(error))
                if self.map_message is not None:raise
                # A freshly booted mapper may not have any graph yet. Record this
                # explicitly; an observed accumulated map is never discarded unsaved.
            finally:
                (evidence/'recovery.json').write_text(json.dumps(metadata,indent=2))
                self.destroy_client(client)
            restart_mapping(args.lock_fd)
        def clear_mapping(self):
            self.mapping_after_stamp=self.get_clock().now().nanoseconds/1e9
            self.latest.pop('map',None);self.latest.pop('odom',None)
            self.map_message=None;self.invalid_streak=0
            self.listener.unregister()
            self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)

    node=Probe()
    try:
        preflight(node)
    except Exception as error:
        node.status('sensor_startup_failed',node.snapshot(),attempted=any(r['recovery_attempted'] for r in node.records))
        print(f'Sensor startup: {error}. Diagnostics: {evidence}',file=sys.stderr,flush=True)
        return 1
    finally:
        node.camera.closed.set();node.destroy_node();rclpy.shutdown()
    print(f'Sensors ready. Diagnostics: {evidence}',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
