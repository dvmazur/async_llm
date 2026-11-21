import json,sys,math
import numpy as np,rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=sys.argv[1],storage_id='sqlite3'),rosbag2_py.ConverterOptions('',''))
types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types()};rows=[];mt=None
angle=lambda q:math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
while reader.has_next():
 topic,data,t=reader.read_next();m=deserialize_message(data,types[topic])
 if topic=='/tf':
  for tf in m.transforms:
   if tf.header.frame_id=='map' and tf.child_frame_id=='odom':mt=tf.transform
 if topic=='/icp/odom' and mt is not None:
  p=m.pose.pose;a=angle(mt.rotation);c,s=math.cos(a),math.sin(a)
  rows.append([t/1e9,p.position.x,p.position.y,mt.translation.x+c*p.position.x-s*p.position.y,mt.translation.y+s*p.position.x+c*p.position.y,a+angle(p.orientation)])
a=np.asarray(rows)
start=a[a[:,0]<a[0,0]+3];end=a[a[:,0]>a[-1,0]-3]
result=dict(samples=len(a),endpoint_averaging_s=3,raw_odom_endpoint_gap_m=float(np.linalg.norm(end[:,1:3].mean(axis=0)-start[:,1:3].mean(axis=0))),slam_endpoint_gap_m=float(np.linalg.norm(end[:,3:5].mean(axis=0)-start[:,3:5].mean(axis=0))),note='Approximate manual return, no external ground truth; latest delivered map->odom, averaging stationary endpoints')
print(json.dumps(result,indent=2))
