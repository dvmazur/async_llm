# No-IMU mapping profile for rover 107

Installed 2026-11-14, no motor commands. Live Qwen and simulation source paths are unchanged. Persistent hardware-side configuration is in the ros container, `/src/packages/main/config/params.yaml` and `/src/packages/main/launch/icp.yaml`; this directory contains the reproducible installer, effective profiles, replay diagnostics and rollback tools. The rover copy is `/opt/rover-mapping-quality`.

The profile enables SLAM scan matching and barycentre, Huber loss, 2 cm occupancy resolution, 1 Hz map publication, 3 cm / 0.04 rad keyframe thresholds, 6 m mapping range, and stricter loop closure acceptance. Existing odom/map/base_link frames and sensor geometry are retained. ICP explicitly estimates floor motion x/y/yaw, uses 2 cm point downsampling, a 0.2 match ratio and 4000-point local reference map. Automatic tracking reset after a single failure is disabled. Missing tracking must be reported as missing, not as trustworthy progress. Point-to-plane ICP remains the live method unless the final benchmark documents otherwise.

Before installation, raw configs were copied to `/opt/rover-mapping-quality/params-before.yaml` and `icp-before.yaml`; an additional copy is at `~/rover-mapping-quality-backup` on the Pi host. The old SLAM graph is at `/opt/rover-mapping-quality/evidence/slam-before-20261114.{data,posegraph}`. The new graph after the manual pass is `slam-quality-after-manual.{data,posegraph}`. The initial distorted graph is not automatically resumed. New mapping configuration persists when the standard container main launch starts after a reboot; it does not depend on a Qwen flag.

Current mapping processes are in the container tmux session `rover-mapping-quality`; both are launched by `run_mapping.sh`. `restart_mapping.sh` locks against active live-agent runs before stopping just the ICP and SLAM executables. It does not restart the camera, hardware bridge, network or container. A pre-existing parent SLAM launch can issue an extra lifecycle activation during a hot replacement; check active state and subsequent operation rather than interpreting a duplicate activation at startup as lost mapping. On the standard fresh boot there is only the main launch.

## Apply / rollback

Never restart mapping during a rover run. New origin invalidates the previous run's coordinate history. Stop the run first. Config and restart commands execute inside `docker exec ros`:

```
python3 /opt/rover-mapping-quality/mapping_quality.py
bash /opt/rover-mapping-quality/restart_mapping.sh
```

Restore exactly the original config and restart the mapping nodes (starts an empty mapping session; the archived graph remains available separately):

```
bash /opt/rover-mapping-quality/restart_mapping.sh --restore
```

After a mapping restart, restart the passive viewer through `rover-qwen.sh --rover 107 view-stop` and `view` so its old coordinate trail is cleared. Ordinary `start` already restarts the viewer and creates fresh action history.

## Evidence and limitations

The 35 s confirmed stationary recording had 341 valid ICP poses, 1.6 cm start-to-end displacement, about 2.9 degrees peak-to-peak yaw variation, and zero z/roll/pitch motion. Accumulated raw odometry path was 1.74 m: summing scan-to-scan jitter is NOT a measurement of real travel. Do not use this number as calibration accuracy.

The manual recording contains about 65 s, 635 lidar scans, 636 valid live odometry poses, and a turn covering about 185 degrees. There were no invalid live poses or automatic resets in this recording. Three-second endpoint averages gave about 21 cm separation in raw ICP and 16 cm after SLAM correction. The physical return point was approximate, so neither is an absolute localization-error measurement.

`compare_icp.py` replays only scan and tf_static on ROS_DOMAIN_ID=219. Neither recorded dynamic transforms nor the live control domain are used; publish_tf is disabled for the comparison nodes. It compares original parameters, updated point-to-plane and updated point-to-point. Reject null/high-covariance tracking-loss messages when computing motion statistics. Results quantify this recording, not all environments.

Raw LD19 messages expose scan_time=0 and time_increment=0. No per-ray motion compensation was enabled using invented timestamps. Fast rotation, repeated/symmetric geometry, glass and objects below the scan plane remain physical limitations. No wheel/IMU fusion or new sensor calibration is claimed. The chosen range/profile targets this small indoor rover, not an arbitrary outdoor vehicle.

Final replay (same manual scan/static-TF recording, separate domain; null poses excluded): baseline 614 valid / 15 invalid messages, updated point-to-plane 629 / 1, updated point-to-point 632 / 1. The retained live method is point-to-plane: both alternatives improve tracking on this recording, but there is insufficient external ground truth to justify switching algorithms based on approximate endpoint error alone. Counts vary slightly with replay scheduling. Do not interpret integrated path jitter as actual distance or count null origin poses as perfect return.

Final ordinary start --dry-run on rover 107: inference_verified; executed_commands=0; accumulated-map projection confirms resolution 0.02 m. Output inside ros: /tmp/rover-qwen-run-20261114T114746-21567.
