import unittest,copy,yaml
from mapping_quality import configure,SLAM,ICP
class ProfileTests(unittest.TestCase):
 def test_changes_only_mapping_and_keeps_node_remaps_and_hardware_parameters(self):
  original={'slam_toolbox':{'ros__parameters':{'use_scan_matching':False,'odom_frame':'odom','scan_topic':'/scan'}},'controller_server':{'ros__parameters':{'controller_frequency':20}}}
  node={'launch':[{'node':{'exec':'icp_odometry','param':[{'name':'publish_tf','value':True}], 'remap':[{'from':'odom','to':'/icp/odom'}]}}]}
  before=copy.deepcopy((original,node));main,launch=configure(original,node)
  self.assertEqual((original,node),before)
  self.assertEqual(main['controller_server'],original['controller_server'])
  self.assertEqual(main['slam_toolbox']['ros__parameters']['odom_frame'],'odom')
  self.assertTrue(main['slam_toolbox']['ros__parameters']['use_scan_matching'])
  params={p['name']:p['value'] for p in launch['launch'][0]['node']['param']}
  self.assertTrue(params['publish_tf']);self.assertEqual(params['Reg/Force3DoF'],'true')
  self.assertEqual(params['Odom/ResetCountdown'],'0')
  self.assertEqual(yaml.safe_load(yaml.safe_dump(main)),main)
  self.assertEqual(configure(main,launch),(main,launch))
 def test_rejects_ambiguous_target(self):
  with self.assertRaises(ValueError):configure({'slam_toolbox':{'ros__parameters':{}}},{'launch':[]})
if __name__=='__main__':unittest.main()
