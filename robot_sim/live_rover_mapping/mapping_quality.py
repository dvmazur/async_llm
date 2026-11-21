"""Reversible no-IMU mapping profile for the floor rover; no motor interfaces."""
import argparse
import json
from pathlib import Path
import shutil
import yaml

SLAM = dict(use_scan_matching=True, use_scan_barycenter=True,
            resolution=.02, min_laser_range=.05, max_laser_range=6.0,
            map_update_interval=1.0, minimum_travel_distance=.03,
            minimum_travel_heading=.04, minimum_time_interval=.15,
            scan_queue_size=1, transform_timeout=.3,
            ceres_loss_function='HuberLoss', loop_match_minimum_chain_size=15,
            loop_match_minimum_response_coarse=.45, loop_match_minimum_response_fine=.65)
ICP = {'Reg/Force3DoF': 'true', 'Odom/ResetCountdown': '0',
       'Icp/CorrespondenceRatio': '0.2', 'Icp/VoxelSize': '0.02',
       'scan_voxel_size': .02, 'scan_range_max': 6.0,
       'OdomF2M/ScanSubtractRadius': '0.02', 'OdomF2M/ScanMaxSize': '4000'}


def configure(main, launch):
    import copy
    main=copy.deepcopy(main);launch=copy.deepcopy(launch)
    main['slam_toolbox']['ros__parameters'].update(SLAM)
    nodes=[item['node'] for item in launch['launch'] if 'node' in item and item['node'].get('exec')=='icp_odometry']
    if len(nodes)!=1:raise ValueError('Expected exactly one ICP node')
    node=nodes[0]
    original={item['name']:item['value'] for item in node.get('param',[])}
    original.update(ICP)
    node['param']=[dict(name=name,value=value) for name,value in original.items()]
    return main,launch


def install(root, dest):
    config=(root/'config/params.yaml').resolve()
    launch=(root/'launch/icp.yaml').resolve()
    dest.mkdir(parents=True,exist_ok=True)
    for path,name in ((config,'params-before.yaml'),(launch,'icp-before.yaml')):
        backup=dest/name
        if not backup.exists():shutil.copy2(path,backup)
    main,newlaunch=configure(yaml.safe_load(config.read_text()),yaml.safe_load(launch.read_text()))
    for value,path in ((main,config),(newlaunch,launch)):
        text=yaml.safe_dump(value,sort_keys=False)
        assert yaml.safe_load(text)==value
        tmp=path.with_suffix(path.suffix+'.quality-new');tmp.write_text(text);tmp.replace(path)
    (dest/'slam.yaml').write_text(yaml.safe_dump({'slam_toolbox':main['slam_toolbox']},sort_keys=False))
    (dest/'icp.yaml').write_text(yaml.safe_dump({'icp_odom':{'ros__parameters':{
        item['name']:item['value'] for item in newlaunch['launch'][0]['node']['param']}}},sort_keys=False))
    (dest/'installation.json').write_text(json.dumps(dict(config=str(config),icp_launch=str(launch),slam=SLAM,icp=ICP),indent=2))
    print('Installed reversible mapping profile in',dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/src/install/main/share/main'))
    parser.add_argument('--dest',type=Path,default=Path('/opt/rover-mapping-quality'))
    parser.add_argument('--restore',action='store_true')
    args=parser.parse_args()
    if args.restore:
        manifest=json.loads((args.dest/'installation.json').read_text())
        for key,name in (('config','params-before.yaml'),('icp_launch','icp-before.yaml')):
            shutil.copy2(args.dest/name,manifest[key])
        print('Original mapping configuration restored; restart mapping nodes to apply')
    else:
        install(args.root,args.dest)
