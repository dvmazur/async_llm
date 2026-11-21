"""Replay measured lidar on an isolated ROS domain; no motor topics."""
import os,sys,subprocess,time,signal,json,math
from pathlib import Path
import yaml,numpy as np,rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
base=Path('/opt/rover-mapping-quality')
profile=yaml.safe_load((base/'icp.yaml').read_text())['icp_odom']['ros__parameters']
children=[];rows={'baseline':[],'plane':[],'points':[]};logs=[]; invalid={name:0 for name in rows}
try:
 for name in rows:
  params=dict(profile)
  if name=='baseline':
   old=yaml.safe_load((base/'icp-before.yaml').read_text())['launch'][0]['node']['param']
   params={item['name']:item['value'] for item in old}
  params.update(publish_tf=False,use_sim_time=True)
  params['Icp/PointToPlane']='false' if name=='points' else 'true'
  file=base/f'offline-{name}.yaml';file.write_text(yaml.safe_dump({'/**':{'ros__parameters':params}}))
  log=(base/f'offline-{name}.log').open('w');logs.append(log)
  children.append(subprocess.Popen(['/opt/ros/jazzy/lib/rtabmap_odom/icp_odometry','--ros-args','--params-file',str(file),'-r',f'__node:=quality_test_{name}','-r','scan:=/scan','-r',f'odom:=/test/{name}/odom'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
 rclpy.init();node=Node('mapping_comparison_recorder')
 def receive(m,name):
  p=m.pose.pose;q=p.orientation
  if m.pose.covariance[0]>=9999 or q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w<.5:
   invalid[name]+=1;return
  rows[name].append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,p.position.x,p.position.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))])
 for name in rows:node.create_subscription(Odometry,f'/test/{name}/odom',lambda m,name=name:receive(m,name),100)
 time.sleep(3)
 playlog=(base/'offline-play.log').open('w');logs.append(playlog)
 player=subprocess.Popen(['ros2','bag','play',sys.argv[1],'--clock','40','--topics','/scan','/tf_static'],stdout=playlog,stderr=subprocess.STDOUT,start_new_session=True);children.append(player)
 deadline=time.monotonic()+90
 while player.poll() is None and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
 for _ in range(30):rclpy.spin_once(node,timeout_sec=.05)
 result={}
 for name,data in rows.items():
  if len(data)<200:raise RuntimeError(f'Insufficient samples: {name}: {len(data)}')
  a=np.array(data);d=np.linalg.norm(np.diff(a[:,1:3],axis=0),axis=1)
  result[name]=dict(count=len(a),invalid_pose_messages=invalid[name],duration_s=float(a[-1,0]-a[0,0]),path_m=float(d.sum()),displacement_m=float(np.linalg.norm(a[-1,1:3]-a[0,1:3])),max_step_m=float(d.max()),yaw_extent_deg=float(np.ptp(np.unwrap(a[:,3]))*180/math.pi))
 (base/'evidence/icp-manual-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
finally:
 for child in reversed(children):
  if child.poll() is None:os.killpg(child.pid,signal.SIGINT)
 for child in children:
  try:child.wait(timeout=8)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 for log in logs:log.close()
