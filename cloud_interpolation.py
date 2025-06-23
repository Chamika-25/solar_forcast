import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def interpolate_cloud_params(e1, e2, t):
    if len(e1) >= 7:
        cx1, cy1, cw1, ch1, crot1, cop1, alt1 = e1
    else:
        cx1, cy1, cw1, ch1, crot1, cop1 = e1
        alt1 = 1.0
    
    if len(e2) >= 7:
        cx2, cy2, cw2, ch2, crot2, cop2, alt2 = e2
    else:
        cx2, cy2, cw2, ch2, crot2, cop2 = e2
        alt2 = 1.0
    
    rot1_rad = crot1
    rot2_rad = crot2
    delta_rot = ((rot2_rad - rot1_rad + np.pi) % (2 * np.pi)) - np.pi
    rot_interp = rot1_rad + t * delta_rot
    alt_interp = alt1 + t * (alt2 - alt1)
    
    return (
        cx1 + t * (cx2 - cx1),
        cy1 + t * (cy2 - cy1),
        cw1 + t * (cw2 - cw1),
        ch1 + t * (ch2 - ch1),
        rot_interp,
        cop1 + t * (cop2 - cop1),
        alt_interp
    )

def create_enhanced_cloud(ellipse_params, alpha_factor, domain_size, area_size_km):
    if len(ellipse_params) >= 7:
        cx, cy, cw, ch, crot, cop, altitude = ellipse_params
    else:
        cx, cy, cw, ch, crot, cop = ellipse_params
        altitude = 1.0
    
    km_x = cx / domain_size * area_size_km
    km_y = cy / domain_size * area_size_km
    width_km = cw / domain_size * area_size_km
    height_km = ch / domain_size * area_size_km
    alpha = min(0.95, cop * 2.0) * alpha_factor
    patches = []
    main_cloud = Ellipse(
        (km_x, km_y), width_km, height_km,
        angle=np.degrees(crot),
        alpha=alpha,
        facecolor="#f0f0f0",
        edgecolor="#e0e0e0",
        linewidth=0.5,
        zorder=5
    )
    patches.append(main_cloud)
    outer_cloud = Ellipse(
        (km_x, km_y), width_km * 1.1, height_km * 1.1,
        angle=np.degrees(crot),
        alpha=alpha * 0.3,
        facecolor="#f8f8f8",
        edgecolor=None,
        linewidth=0,
        zorder=4
    )
    patches.append(outer_cloud)
    offset_x = width_km * 0.15 * np.cos(crot + np.pi/4)
    offset_y = height_km * 0.15 * np.sin(crot + np.pi/4)
    highlight = Ellipse(
        (km_x - offset_x, km_y - offset_y), 
        width_km * 0.4, height_km * 0.4,
        angle=np.degrees(crot - 0.2),
        alpha=alpha * 0.8,
        facecolor="#ffffff",
        edgecolor=None,
        linewidth=0,
        zorder=6
    )
    patches.append(highlight)
    return patches

def interpolate_clouds(source_ellipses, target_ellipses, t, domain_size, area_size_km):
    if len(source_ellipses) == 0:
        return [p for e in target_ellipses for p in create_enhanced_cloud(e, t, domain_size, area_size_km)]
    
    if len(target_ellipses) == 0:
        return [p for e in source_ellipses for p in create_enhanced_cloud(e, 1-t, domain_size, area_size_km)]
    
    cloud_patches = []
    
    if len(source_ellipses) == 1 and len(target_ellipses) == 1:
        interpolated = interpolate_cloud_params(source_ellipses[0], target_ellipses[0], t)
        patches = create_enhanced_cloud(interpolated, 1.0, domain_size, area_size_km)
        cloud_patches.extend(patches)
        return cloud_patches
    
    for i, e1 in enumerate(source_ellipses):
        if i < len(target_ellipses):
            e2 = target_ellipses[i]
            interpolated = interpolate_cloud_params(e1, e2, t)
            patches = create_enhanced_cloud(interpolated, 1.0, domain_size, area_size_km)
            cloud_patches.extend(patches)
        else:
            fade_factor = max(0, 1 - t*2)
            patches = create_enhanced_cloud(e1, fade_factor, domain_size, area_size_km)
            cloud_patches.extend(patches)
    
    for i, e2 in enumerate(target_ellipses):
        if i >= len(source_ellipses):
            fade_factor = min(1, t*2)
            patches = create_enhanced_cloud(e2, fade_factor, domain_size, area_size_km)
            cloud_patches.extend(patches)
    
    return cloud_patches