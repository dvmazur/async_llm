import sys,json,math
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

reader=rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=sys.argv[1],storage_id='sqlite3'),rosbag2_py.ConverterOptions('',''))
types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types()}
odom=[];invalid_odom=0;scans=[];tf={};counts={}
def angle(q):return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
while reader.has_next():
 topic,data,t=reader.read_next();m=deserialize_message(data,types[topic]);counts[topic]=counts.get(topic,0)+1
 if topic=='/icp/odom':
  p=m.pose.pose;q=p.orientation
  if m.pose.covariance[0]>=9999 or q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w<.5:
   invalid_odom+=1;continue
  odom.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,p.position.x,p.position.y,p.position.z,angle(p.orientation),p.orientation.x,p.orientation.y])
 if topic=='/scan':
  r=np.asarray(m.ranges);valid=np.isfinite(r)&(r>=m.range_min)&(r<=m.range_max)
  scans.append(dict(stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,beams=len(r),valid=int(valid.sum()),range_min=m.range_min,range_max=m.range_max,near=float(r[valid].min()) if valid.any() else None,scan_time=m.scan_time,time_increment=m.time_increment))
 if topic=='/tf':
  for x in m.transforms:tf[x.header.frame_id+' -> '+x.child_frame_id]=tf.get(x.header.frame_id+' -> '+x.child_frame_id,0)+1
s=dict(messages=counts,invalid_odom=invalid_odom,tf_edges=tf,scans=scans[:2])
if odom:
 a=np.asarray(odom);d=np.linalg.norm(np.diff(a[:,1:3],axis=0),axis=1);yaw=np.unwrap(a[:,4])
 s['odom']=dict(count=len(a),duration_s=float(a[-1,0]-a[0,0]),path_m=float(d.sum()),displacement_m=float(np.linalg.norm(a[-1,1:3]-a[0,1:3])),max_step_m=float(d.max()),z_extent_m=float(np.ptp(a[:,3])),yaw_extent_deg=float(np.ptp(yaw)*180/math.pi),max_qxy=float(np.abs(a[:,5:7]).max()),max_gap_s=float(np.diff(a[:,0]).max()))
print(json.dumps(s,indent=2))
