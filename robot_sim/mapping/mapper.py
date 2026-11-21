"""Live sensor-derived top-down maps. No simulator pose, scene or map input."""
import argparse
from collections import deque
import copy
import json
import math
from pathlib import Path
import time

import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
import numpy as np
from PIL import Image as PILImage,ImageDraw,ImageFont
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan,Image,CameraInfo
from nav_msgs.msg import OccupancyGrid,Odometry
from std_msgs.msg import String
from tf2_ros import Buffer,TransformListener
import yaml

from robot_sim.mapping.core import LocalGrid,ClusterTracker,filter_scan,transform_points,project,occupancy_to_pgm


def stamp(msg):
    return msg.header.stamp.sec+msg.header.stamp.nanosec/1e9


def transform(tf,points):
    t,q=tf.transform.translation,tf.transform.rotation
    return transform_points(points,[t.x,t.y,t.z],[q.x,q.y,q.z,q.w])


class Mapper(Node):
    def __init__(self,args):
        super().__init__("sensor_map_builder")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time",value=True)])
        self.args=args;self.grid=LocalGrid();self.tracker=ClusterTracker()
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.cameras=deque(maxlen=8);self.calibration=None;self.color_pending=deque(maxlen=12)
        self.map=None;self.odom=None;self.tracks=[];self.path=[];self.last_scan=-1
        self.stats={"scans":0,"tf_dropped":0,"camera_pairs":0,"colored_points":0,"camera_unmatched":0,
                    "moving_track_frames":0,"global_maps":0}
        self.started=time.monotonic();self.frames=[];self.last_publish=-1;self.last_snapshot=-1
        self.log=(args.output/"mapping.jsonl").open("w")
        self.frame_log=(args.output/"map_frames.jsonl").open("w")
        (args.output/"map_frames").mkdir(exist_ok=True)
        latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.local_pub=self.create_publisher(OccupancyGrid,"/mapping/dynamic_grid",latched)
        self.risk_pub=self.create_publisher(OccupancyGrid,"/mapping/costmap",latched)
        self.track_pub=self.create_publisher(String,"/mapping/tracks",10)
        self.scan_pub=self.create_publisher(LaserScan,"/scan/slam",qos_profile_sensor_data)
        self.create_subscription(LaserScan,"/scan",self.scan,qos_profile_sensor_data)
        self.create_subscription(Image,"/camera/image_raw",self.image,qos_profile_sensor_data)
        self.create_subscription(CameraInfo,"/camera/camera_info",self.camera_info,10)
        self.create_subscription(Odometry,"/odom",self.odometry,10)
        self.create_subscription(OccupancyGrid,"/map",self.global_map,latched)
        self.create_timer(.5,self.publish)
        (args.output/"mapping_inputs.json").write_text(json.dumps({
            "mapper":["/scan","/camera/image_raw","/camera/camera_info","/odom","/tf","/tf_static","/map","/clock"],
            "ekf":["/wheel/odometry","/imu/data","/tf_static","/clock"],
            "slam":["/scan/slam","/tf","/tf_static","/clock"],
            "ground_truth_used":False,"existing_map_loaded":False,
            "camera_role":"calibrated RGB annotation of lidar hits; no monocular depth"},indent=2))

    def camera_info(self,msg):
        self.calibration=list(msg.k)

    def image(self,msg):
        array=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)
        rgb=array[:,:msg.width*3].reshape(msg.height,msg.width,3).copy()
        self.cameras.append((stamp(msg),rgb,msg.header.frame_id))
        self.colorize_pending()

    def odometry(self,msg):
        self.odom=msg
        p=msg.pose.pose.position
        if getattr(self.args,"rolling",False):self.grid.recenter((p.x,p.y))
        self.path.append([p.x,p.y])
        if len(self.path)>10000:self.path=self.path[::2]

    def global_map(self,msg):
        self.map=msg;self.stats["global_maps"]+=1

    def scan(self,msg):
        now=stamp(msg)
        if now-self.last_scan<.19:return
        self.last_scan=now
        ranges=filter_scan(msg.ranges)
        filtered=copy.deepcopy(msg);filtered.ranges=ranges.astype(np.float32).tolist()
        self.scan_pub.publish(filtered)
        try:
            tf=self.buffer.lookup_transform("odom",msg.header.frame_id,Time.from_msg(msg.header.stamp))
        except Exception:
            self.stats["tf_dropped"]+=1;return
        angles=msg.angle_min+np.arange(len(ranges))*msg.angle_increment
        valid=((np.isfinite(ranges)&(ranges>=msg.range_min))|np.isposinf(ranges))
        rays=np.minimum(ranges[valid],msg.range_max)
        hits=np.isfinite(ranges[valid])&(ranges[valid]<msg.range_max-.01)
        local=np.column_stack([rays*np.cos(angles[valid]),rays*np.sin(angles[valid]),np.zeros(len(rays))])
        world=transform(tf,local)
        sensor=transform(tf,np.zeros((1,3)))[0]
        self.grid.update(sensor[:2],world[:,:2],hits,now)
        self.stats["scans"]+=1
        hit_local=local[hits];hit_world=world[hits]
        self.color_pending.append((now,msg.header.frame_id,msg.header.stamp,hit_local,hit_world))
        self.colorize_pending()
        clusters=[];group=[]
        for point in hit_world[:,:2]:
            if group and np.linalg.norm(point-group[-1])>.18:
                if len(group)>=4 and np.linalg.norm(group[0]-group[-1])<.5:clusters.append(np.mean(group,axis=0))
                group=[]
            group.append(point)
        if len(group)>=4 and np.linalg.norm(group[0]-group[-1])<.5:clusters.append(np.mean(group,axis=0))
        self.tracks=self.tracker.update(clusters,now)
        self.stats["moving_track_frames"]+=int(any(t["moving"] for t in self.tracks))

    def colorize_pending(self):
        # Wait briefly for the adjacent camera frame instead of pairing with an
        # old image merely because RGB delivery latency exceeds lidar latency.
        current=self.get_clock().now().nanoseconds/1e9
        while self.color_pending and current-self.color_pending[0][0]>.13:
            now,frame,scan_stamp,hit_local,hit_world=self.color_pending.popleft()
            if self.cameras and self.calibration:
                selected=min(self.cameras,key=lambda frame:abs(frame[0]-now))
                if abs(selected[0]-now)<=.12:
                    try:
                        cam_time=Time(seconds=int(selected[0]),nanoseconds=int((selected[0]%1)*1e9))
                        camera_tf=self.buffer.lookup_transform_full(selected[2],cam_time,
                            frame,Time.from_msg(scan_stamp),"odom")
                        cam_points=transform(camera_tf,hit_local)
                        uv,in_frame=project(cam_points,self.calibration,selected[1].shape[1],selected[1].shape[0])
                        colors=[];points=[]
                        # Nearest lidar surface wins in a coarse image column.
                        columns={}
                        for i in np.flatnonzero(in_frame):
                            key=int(uv[i,0]//4)
                            if key not in columns or cam_points[i,2]<cam_points[columns[key],2]:columns[key]=i
                        for i in columns.values():
                            u,v=np.rint(uv[i]).astype(int)
                            colors.append(np.median(selected[1][v-1:v+2,u-1:u+2],axis=(0,1)))
                            points.append(hit_world[i,:2])
                        if points:self.grid.colorize(np.array(points),np.array(colors),now)
                        self.stats["camera_pairs"]+=1;self.stats["colored_points"]+=len(points)
                    except Exception:
                        self.stats["camera_unmatched"]+=1
                else:self.stats["camera_unmatched"]+=1

    def occupancy_message(self,values,now):
        msg=OccupancyGrid();msg.header.frame_id="odom";msg.header.stamp=self.get_clock().now().to_msg()
        msg.info.resolution=self.grid.resolution;msg.info.width=msg.info.height=self.grid.width
        msg.info.origin.position.x,msg.info.origin.position.y=self.grid.origin.tolist()
        msg.info.origin.orientation.w=1.
        msg.data=values.astype(np.int8).ravel().tolist()
        return msg

    def publish(self):
        now=self.get_clock().now().nanoseconds/1e9
        if now<=0:return
        values=self.grid.occupancy(now)
        # Inflate occupied cells for a 13 cm wide robot plus uncertainty margin.
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        inflated=cv2.dilate((values>=65).astype(np.uint8),kernel)>0
        cost=np.where(values<0,-1,0).astype(np.int16)
        cost[inflated]=100
        self.local_pub.publish(self.occupancy_message(values,now))
        self.risk_pub.publish(self.occupancy_message(cost,now))
        self.track_pub.publish(String(data=json.dumps({"sim_time":now,"tracks":self.tracks})))
        self.log.write(json.dumps({"sim_time":now,**self.stats,"known_cells":int(np.sum(values>=0)),
                   "occupied_cells":int(np.sum(values>=65)),"tracks":self.tracks})+"\n");self.log.flush()
        if now-self.last_snapshot>=.95 and not getattr(self.args,"no_preview",False):
            self.last_snapshot=now
            self.snapshot(values,cost,now)
        self.save_grids(values,now)

    def save_grids(self,values,now):
        np.savez_compressed(self.args.output/"local_map.npz",occupancy=values,
                            last_seen=self.grid.seen,resolution=self.grid.resolution,origin=self.grid.origin,stamp=now)
        if self.map is None:return
        grid=np.asarray(self.map.data,dtype=np.int16).reshape(self.map.info.height,self.map.info.width)
        pixels=occupancy_to_pgm(grid)
        np.savez_compressed(self.args.output/"global_map_data.npz",occupancy=grid)
        PILImage.fromarray(np.flipud(pixels)).save(self.args.output/"global_map.pgm")
        origin=self.map.info.origin
        yaw=2*math.atan2(origin.orientation.z,origin.orientation.w)
        (self.args.output/"global_map.yaml").write_text(yaml.safe_dump(
            {"image":"global_map.pgm","resolution":float(self.map.info.resolution),
             "origin":[origin.position.x,origin.position.y,yaw],"negate":0,
             "occupied_thresh":.65,"free_thresh":.35,"mode":"trinary"}))

    def snapshot(self,values,cost,now):
        canvas=PILImage.new("RGB",(1280,800),"#101923");draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
        small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
        draw.text((20,12),f"Sensor-only mapping / t={now:.1f}s / 1 CPU, 0 GPU",font=font,fill="white")
        draw.text((20,42),"GLOBAL SLAM: lidar + fused wheel/IMU pose",font=font,fill="#8ce2c2")
        draw.text((655,42),"LIVE LOCAL MAP: rays + RGB lidar annotations",font=font,fill="#8ce2c2")
        # Fixed local view is a crop around current odometry position, no scene bounds.
        robot=np.array([self.odom.pose.pose.position.x,self.odom.pose.pose.position.y]) if self.odom else np.zeros(2)
        center,_=self.grid.index(robot[None,:]);cx,cy=center[0]
        span=80;x0=cx-span;y0=cy-span
        gray=np.where(values<0,105,255*(1-values/100)).clip(0,255).astype(np.uint8)
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        rgb[(cost==100)&(values<65)]=[188,132,77]
        fresh=(now-self.grid.rgb_seen<1.5)&(values>=55)
        rgb[fresh]=self.grid.rgb[fresh].astype(np.uint8)
        full=PILImage.fromarray(np.flipud(rgb))
        tile=full.crop((x0,self.grid.width-(y0+span*2),x0+span*2,self.grid.width-y0)).resize((600,600),PILImage.Resampling.NEAREST)
        canvas.paste(tile,(655,75))
        local=lambda xy:(655+(xy[0]-self.grid.origin[0]-x0*self.grid.resolution)/self.grid.resolution*600/(2*span),
                          675-(xy[1]-self.grid.origin[1]-y0*self.grid.resolution)/self.grid.resolution*600/(2*span))
        if len(self.path)>1:draw.line([local(p) for p in self.path],fill="#4a82ff",width=2)
        if self.odom:
            q=self.odom.pose.pose.orientation;yaw=2*math.atan2(q.z,q.w)
            p=local(robot)
            draw.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill="#00d8ff")
            draw.line([p,(p[0]+20*math.cos(yaw),p[1]-20*math.sin(yaw))],fill="#00d8ff",width=3)
            covariance=self.odom.pose.covariance
            rx=min(100,2*math.sqrt(max(0,covariance[0]))/self.grid.resolution*600/(2*span))
            ry=min(100,2*math.sqrt(max(0,covariance[7]))/self.grid.resolution*600/(2*span))
            draw.ellipse((p[0]-rx,p[1]-ry,p[0]+rx,p[1]+ry),outline="#00b4db",width=1)
        for t in self.tracks:
            if not t["moving"] or t["age_s"]>.4:continue
            a=local(t["xy"]);b=local(np.array(t["xy"])+np.array(t["velocity"]))
            draw.ellipse((a[0]-6,a[1]-6,a[0]+6,a[1]+6),outline="#ff00a8",width=2)
            draw.line([a,b],fill="#ff00a8",width=3)
        if self.map:
            grid=np.asarray(self.map.data).reshape(self.map.info.height,self.map.info.width)
            gray=np.where(grid<0,105,255*(1-grid/100)).clip(0,255).astype(np.uint8)
            im=PILImage.fromarray(np.flipud(gray)).convert("RGB")
            scale=min(600/im.width,600/im.height)
            im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),PILImage.Resampling.NEAREST)
            px=20+(600-im.width)//2;py=75+(600-im.height)//2;canvas.paste(im,(px,py))
            try:
                tf=self.buffer.lookup_transform("map","odom",Time())
                path3=transform(tf,np.column_stack([np.asarray(self.path),np.zeros(len(self.path))]))
                origin=self.map.info.origin.position
                point=lambda p:(px+(p[0]-origin.x)/self.map.info.resolution*im.width/self.map.info.width,
                                py+im.height-(p[1]-origin.y)/self.map.info.resolution*im.height/self.map.info.height)
                if len(path3)>1:draw.line([point(p) for p in path3],fill="#2288ee",width=2)
                p=point(path3[-1]);draw.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill="#00b4ee")
            except Exception:pass
        else:draw.text((40,100),"Waiting for SLAM map...",font=font,fill="white")
        draw.text((20,690),"Gray: unknown / white: observed free / black: occupied",font=small,fill="white")
        draw.text((20,715),"Brown: clearance margin / magenta: estimated moving clusters",font=small,fill="white")
        draw.text((20,740),f"RGB pairs {self.stats['camera_pairs']} / scans {self.stats['scans']} / global updates {self.stats['global_maps']}",font=small,fill="#8ce2c2")
        draw.text((20,767),"No world geometry, simulator pose or overhead camera is used by the mapper.",font=small,fill="#b4c5d7")
        if self.cameras:
            front=PILImage.fromarray(self.cameras[-1][1]);front.thumbnail((160,120))
            canvas.paste(front,(1090,650))
        filename=f"map_frames/{len(self.frames):05d}.png"
        canvas.save(self.args.output/filename)
        temp=self.args.output/"live.tmp.png";canvas.save(temp);temp.replace(self.args.output/"live.png")
        record={"file":filename,"sim_time":now};self.frames.append(record)
        self.frame_log.write(json.dumps(record)+"\n");self.frame_log.flush()


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rolling",action="store_true",help="Keep the dynamic grid around the rover on long routes")
    p.add_argument("--no-preview",action="store_true",help="Let the candidate agent record the composite view")
    args=p.parse_args();rclpy.init();node=Mapper(args)
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        (args.output/"mapper_summary.json").write_text(json.dumps(node.stats,indent=2))
        node.log.close();node.frame_log.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=="__main__":main()
