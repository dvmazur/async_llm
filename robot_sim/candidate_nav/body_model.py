"""Descriptive live-rover dimensions; does not filter or execute actions."""
import math

# Approximate external dimensions supplied by the operator, not a calibration.
LENGTH_M = 0.20
WIDTH_M = 0.12


def body_description():
    from .live_trajectory import HALF_LENGTH, HALF_WIDTH, HARD_GAP
    return {
        'shape': 'rectangle',
        'approximate_physical_size_m': {'length': LENGTH_M, 'width': WIDTH_M},
        'dimension_source': 'operator estimate; external protrusions not measured',
        'planning_footprint_m': {'length': 2*HALF_LENGTH, 'width': 2*HALF_WIDTH},
        'planning_half_extents_m': {'front': HALF_LENGTH, 'rear': HALF_LENGTH,
                                    'left': HALF_WIDTH, 'right': HALF_WIDTH},
        'planning_corner_radius_m': round(math.hypot(HALF_LENGTH, HALF_WIDTH), 4),
        'swept_scan_margin_m': HARD_GAP,
        'reference': 'planning rectangle centred at the physical base_link pose; x forward, y left',
        'calibration': 'body offsets from camera/lidar and true rotation centre not calibrated',
    }


BODY_RULES = """The rover has a finite rectangular body, not a point or a camera ray.
Read rover_body: approximate physical dimensions are 20 cm long and 12 cm wide; the existing
planning footprint is a conservative 20 by 16 cm rectangle. Use its front/rear/side extents
and measured free space for motion, not the visible centre line alone. A visible gap or light
through objects does not establish a traversable opening: the entire body must fit throughout
the route with clearance for uncertainty. Image pixels alone do not establish metric gap width.
During in-place turns the front/rear corners sweep outward; a path clear for straight travel
may lack space to rotate. Check the whole swept body, including corners, for each planned step
and intermediate orientation, not only the final pose. Forward and reverse both move the full
body; the front camera does not verify rear clearance. Use measured map/scan evidence, not
absence of an obstacle in camera view, to justify unseen space. Keep clearances to the body
boundary distinct from distances to base_footprint, camera or lidar; their offsets are not
calibrated, so do not claim centimetre precision from an image guess. If fit is uncertain, use
observations or a justified short repositioning to resolve it. These principles apply to any
instruction and remain relevant when collision_protection is disabled; they are planning
context, not an additional action veto.
"""
