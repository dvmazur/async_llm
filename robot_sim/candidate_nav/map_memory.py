"""Accumulated SLAM context for live planning, independent of local motion guards."""
import math

MAP_MEMORY_RULES = """Use two distinct maps when accumulated_map is available. IMAGE 2 is LOCAL recent
measured geometry, used for immediate body clearance, candidate letters and proposal pixel
coordinates. IMAGE 3 is ACCUMULATED SLAM: the explored layout over the SLAM session, current
pose and travelled path, without the local eight-second expiry. A yellow outline marks the
local map extent. Read maps_context for each projection, metre scale, frame, age and availability;
the two images can have different scales, origins and orientations. Use accumulated geometry
and the travel trail to relate earlier places and choose a useful route or return direction.
The trail records visits, not proof that every branch was explored or is a dead end. Persistent
free space is historical evidence, not current clearance: recent local map/scan observations
win near the rover. Unknown space is not free. A stale or missing SLAM update is not new evidence.
Do not transfer pixel coordinates or candidate letters from one map to the other. Previous
camera, when supplied, is separately labelled by image_roles; it is not a map. These maps
contain no hidden target or prescribed exploration strategy.
"""


def accumulated_view(grid, pose, path, local_grid, now, size=512):
    """Render the entire available SLAM extent; never substitute a local scan."""
    import numpy as np
    from PIL import ImageDraw
    from .render import render_map, pixel, font
    if grid is None:
        return None, {'available': False, 'reason': 'no_accumulated_slam_received'}
    if grid.frame != local_grid.frame:
        return None, {'available': False, 'reason': 'map_frame_mismatch',
                      'frame': grid.frame, 'local_frame': local_grid.frame}
    if not grid.values.size or not math.isfinite(grid.resolution) or grid.resolution <= 0:
        return None, {'available': False, 'reason': 'invalid_slam_geometry'}
    image, projection = render_map(grid, pose, path=path, size=size)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size, 32), fill=(20,28,39))
    draw.text((12,7), f'ACCUMULATED SLAM | v{grid.version}', font=font(15), fill='white')
    height, width = local_grid.values.shape
    c,s = math.cos(local_grid.yaw),math.sin(local_grid.yaw)
    corners = [(local_grid.origin[0]+c*x-s*y,local_grid.origin[1]+s*x+c*y)
               for x,y in ((0,0),(width*local_grid.resolution,0),
                           (width*local_grid.resolution,height*local_grid.resolution),
                           (0,height*local_grid.resolution))]
    outline = [pixel(xy,projection) for xy in corners]
    draw.line(outline+[outline[0]], fill=(255,218,64), width=2)
    age = now-grid.stamp
    return image, {
        'available': True, 'source': 'slam_toolbox /map', 'scope': 'SLAM session; may predate this run',
        'ttl_s': None, 'version': grid.version, 'stamp': grid.stamp,
        'age_s': round(age,3), 'recent_update': -.15 <= age < 5,
        'per_cell_age': 'not supplied by SLAM; old free cells are not current clearance',
        'frame': grid.frame, 'projection': projection,
        'metres_per_image_pixel': round(grid.resolution/projection['scale'],5),
        'extent_m': [grid.values.shape[1]*grid.resolution, grid.values.shape[0]*grid.resolution],
        'known_cells': int(np.count_nonzero(grid.values>=0)),
        'rover_pixel': list(pixel(pose[:2],projection)),
        'local_extent_world': [list(xy) for xy in corners],
    }
