import math
import numpy as np
from scipy.interpolate import CubicSpline
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================
# Define TOTAL_POINTS (145 data points, one per 5-minute interval)
# ==========================
TOTAL_POINTS = 145

# ==========================
# 1) CUSTOM BASELINE DAILY GENERATION DATA (145 points each)
# ==========================
# Replace the placeholder arrays below with your full 145-point data if needed.
# (If any array isn’t exactly 145 points, it will be resampled below.)

# Panel A baseline data (location A: [8.3,4.2])
custom_baseline_A = np.array([
    0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
    0.4, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 1.0,
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.6, 1.7, 1.9, 2.0, 2.1, 2.2,
    2.3, 2.4, 2.5, 2.7, 2.8, 2.9, 3.0, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8,
    3.9, 4.0, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.0, 5.1,
    5.2, 5.2, 5.3, 5.3, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.4, 5.3,
    5.3, 5.2, 5.2, 5.1, 5.0, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3,
    4.2, 4.0, 3.9, 3.8, 3.7, 3.5, 3.4, 3.3, 3.2, 3.0, 2.9, 2.8, 2.7,
    2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.7, 1.6, 1.6, 1.5, 1.4, 1.3,
    1.2, 1.1, 1.0, 1.0, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5,
    0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1,
    0.1, 0.1, 0.1
])

# Panel D baseline data (location D: [9.2,0.8])
custom_baseline_D = np.array([
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4,
    0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.6, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.1, 1.1, 1.2,
    1.3, 1.4, 1.5, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
    2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.7, 3.8, 3.9, 4.0, 4.1, 4.1, 4.2,
    4.3, 4.3, 4.4, 4.4, 4.4, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.4, 4.4, 4.4,
    4.3, 4.3, 4.2, 4.1, 4.1, 4.0, 3.9, 3.8, 3.7, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1,
    3.0, 2.9, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.5,
    1.4, 1.3, 1.2, 1.1, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.5, 0.5,
    0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1
])

# Panel K baseline data (location K: [0.6,6.3])
custom_baseline_K = np.array([
    0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.5,
    0.6, 0.6, 0.7, 0.8, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
    1.9, 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.9, 3.0, 3.2, 3.3, 3.5, 3.6, 3.8, 4.0, 4.1,
    4.3, 4.4, 4.6, 4.7, 4.9, 5.1, 5.2, 5.3, 5.5, 5.6, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3,
    6.4, 6.5, 6.5, 6.6, 6.6, 6.7, 6.7, 6.7, 6.8, 6.8, 6.7, 6.7, 6.7, 6.6, 6.6, 6.5,
    6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.6, 5.5, 5.3, 5.2, 5.1, 4.9, 4.7, 4.6,
    4.4, 4.3, 4.1, 4.0, 3.8, 3.6, 3.5, 3.3, 3.2, 3.0, 2.9, 2.7, 2.6, 2.4, 2.3, 2.2,
    2.1, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 1.0, 0.9, 0.8, 0.8, 0.7,
    0.6, 0.6, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1,
    0.1
])

# Panel J baseline data (location J: [3.9,2.0])
custom_baseline_J = np.array([
    0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8,
    1.9, 2.1, 2.3, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.9, 4.1, 4.4, 4.6, 4.9, 5.2, 5.5, 5.7, 6.0, 6.3, 6.7, 7.0,
    7.3, 7.6, 7.9, 8.2, 8.6, 8.9, 9.2, 9.5, 9.8, 10.1, 10.4, 10.7, 11.0, 11.2, 11.5, 11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 12.9,
    13.1, 13.2, 13.3, 13.4, 13.4, 13.5, 13.5, 13.5, 13.5, 13.4, 13.4, 13.3, 13.2, 13.1, 12.9, 12.8, 12.6, 12.4, 12.2, 12.0,
    11.8, 11.5, 11.2, 11.0, 10.7, 10.4, 10.1, 9.8, 9.5, 9.2, 8.9, 8.6, 8.2, 7.9, 7.6, 7.3, 7.0, 6.7, 6.3, 6.0,
    5.7, 5.5, 5.2, 4.9, 4.6, 4.4, 4.1, 3.9, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.3, 2.1,
    1.9, 1.8, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.5,
    0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2
])

# Panel L baseline data (location L: [3.1,4.9])
custom_baseline_L = np.array([
    0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6,
    0.6, 0.7, 0.7, 0.8, 0.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
    1.7, 1.9, 2.0, 2.1, 2.3, 2.4, 2.6, 2.7, 2.9, 3.1, 3.3, 3.4, 3.6,
    3.8, 4.0, 4.2, 4.4, 4.6, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9, 6.1, 6.3,
    6.5, 6.7, 6.9, 7.1, 7.3, 7.5, 7.7, 7.8, 8.0, 8.1, 8.3, 8.4, 8.5,
    8.6, 8.7, 8.8, 8.9, 8.9, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 8.9, 8.9,
    8.8, 8.7, 8.6, 8.5, 8.4, 8.3, 8.1, 8.0, 7.8, 7.7, 7.5, 7.3, 7.1,
    6.9, 6.7, 6.5, 6.3, 6.1, 5.9, 5.7, 5.5, 5.3, 5.1, 4.9, 4.6, 4.4,
    4.2, 4.0, 3.8, 3.6, 3.4, 3.3, 3.1, 2.9, 2.7, 2.6, 2.4, 2.3, 2.1,
    2.0, 1.9, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.9, 0.8,
    0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.2,
    0.2, 0.2, 0.2, 0.2, 0.1, 0.1
])

# Panel G baseline data (location G: [0.6,6.3])
custom_baseline_G = np.array([
    0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.8, 0.9,
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.9, 2.0, 2.2, 2.4,
    2.6, 2.8, 3.0, 3.2, 3.5, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2,
    5.5, 5.8, 6.2, 6.5, 6.9, 7.3, 7.7, 8.1, 8.5, 8.9, 9.3,
    9.7, 10.1, 10.6, 11.0, 11.4, 11.8, 12.2, 12.7, 13.1, 13.5, 13.9,
    14.3, 14.6, 15.0, 15.3, 15.7, 16.0, 16.3, 16.6, 16.8, 17.0, 17.2,
    17.4, 17.6, 17.7, 17.8, 17.9, 18.0, 18.0, 18.0, 18.0, 17.9, 17.8,
    17.7, 17.6, 17.4, 17.2, 17.0, 16.8, 16.6, 16.3, 16.0, 15.7, 15.3,
    15.0, 14.6, 14.3, 13.9, 13.5, 13.1, 12.7, 12.2, 11.8, 11.4, 11.0,
    10.6, 10.1, 9.7, 9.3, 8.9, 8.5, 8.1, 7.7, 7.3, 6.9, 6.5,
    6.2, 5.8, 5.5, 5.2, 4.9, 4.6, 4.3, 4.0, 3.7, 3.5, 3.2,
    3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.9, 1.7, 1.6, 1.5, 1.3,
    1.2, 1.1, 1.0, 0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4,
    0.4, 0.3
])

# ==========================
# 2) SIMULATION SETTINGS
# ==========================
TOTAL_FRAMES = TOTAL_POINTS  # 145 frames (each representing 5 minutes)
INTERVAL_MS = 200  # 200 ms per frame

# ==========================
# 3) PANEL LOCATIONS & NAMES
# ==========================
# Default coordinates for 20 panels (in km)
default_coords = [
    (8.3, 4.2),   # A
    (1.6, 9.8),   # B
    (6.7, 5.1),   # C
    (9.2, 0.8),   # D
    (4.5, 7.1),   # E
    (7.3, 8.6),   # F
    (0.6, 6.3),   # G  -> custom for G below
    (3.9, 2.0),   # H
    (5.8, 8.4),   # I
    (3.1, 4.9),   # J  -> custom for J below
    (8.2, 9.9),   # K  -> custom for K below
    (5.7, 5.6),   # L  -> custom for L below
    (2.9, 0.3),   # M
    (6.4, 7.8),   # N
    (9.3, 3.9),   # O
    (5.1, 6.9),   # P
    (7.0, 9.2),   # Q
    (1.2, 4.6),   # R
    (9.5, 2.5),   # S
    (5.1, 1.2)    # T
]
# Override coordinates for panels with custom baselines.
custom_coords = {
    "A": (8.3, 4.2),   # Panel A
    "D": (9.2, 0.8),   # Panel D
    "G": (0.6, 6.3),   # Panel G
    "J": (3.9, 2.0),   # Panel J
    "K": (0.6, 6.3),   # Panel K
    "L": (3.1, 4.9)    # Panel L
}
panel_names = [chr(ord('A') + i) for i in range(len(default_coords))]
panels = []
for i, pname in enumerate(panel_names):
    coord = custom_coords.get(pname, default_coords[i])
    panels.append({
        'name': pname,
        'x_km': coord[0],
        'y_km': coord[1]
    })

# ==========================
# 4) SIMULATION BASELINE & CLOUD FACTOR SETUP
# ==========================
# For panels with custom baselines, use their arrays; for all others, use the common baseline.
baseline_data_common = np.array([
    0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 
    1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.1, 3.7, 4.3, 5.0, 5.7, 6.5, 7.4, 8.3, 9.3, 10.4, 11.5, 12.7,
    13.9, 15.2, 16.5, 17.8, 19.2, 20.6, 22.0, 23.5, 25.0, 26.5, 28.0, 29.4, 30.8, 32.1, 33.3, 34.4,
    35.4, 36.3, 37.0, 37.6, 38.0, 38.3, 38.5, 38.6, 38.6, 38.5, 38.3, 38.0, 37.6, 37.0, 36.3, 35.4,
    34.4, 33.3, 32.1, 30.8, 29.4, 28.0, 26.5, 25.0, 23.5, 22.0, 20.6, 19.2, 17.8, 16.5, 15.2, 13.9,
    12.7, 11.5, 10.4, 9.3, 8.3, 7.4, 6.5, 5.7, 5.0, 4.3, 3.7, 3.1, 2.6, 2.2, 1.8, 1.5,
    1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1
])
baseline_data_dict = {}
for pname in panel_names:
    if pname == "A":
        baseline_data_dict[pname] = custom_baseline_A
    elif pname == "D":
        baseline_data_dict[pname] = custom_baseline_D
    elif pname == "K":
        baseline_data_dict[pname] = custom_baseline_K
    elif pname == "J":
        baseline_data_dict[pname] = custom_baseline_J
    elif pname == "L":
        baseline_data_dict[pname] = custom_baseline_L
    elif pname == "G":
        baseline_data_dict[pname] = custom_baseline_G
    else:
        baseline_data_dict[pname] = baseline_data_common
    # Resample if needed:
    if len(baseline_data_dict[pname]) != TOTAL_POINTS:
        x_old = np.linspace(0, len(baseline_data_dict[pname]) - 1, len(baseline_data_dict[pname]))
        x_new = np.linspace(0, len(baseline_data_dict[pname]) - 1, TOTAL_POINTS)
        baseline_data_dict[pname] = np.interp(x_new, x_old, baseline_data_dict[pname])
        
# Build cubic splines for each panel.
time_idx = np.linspace(0, TOTAL_POINTS - 1, TOTAL_POINTS)
spline_dict = {}
for pname in panel_names:
    spline_dict[pname] = CubicSpline(time_idx, baseline_data_dict[pname])

# Define individual cloud factors.
custom_cloud_factors = {
    "A": 0.7,
    "D": 0.65,
    "K": 0.55,
    "J": 0.6,
    "L": 0.6,
    "G": 0.55
}
cloud_factor_dict = {}
for pname in panel_names:
    if pname in custom_cloud_factors:
        cloud_factor_dict[pname] = custom_cloud_factors[pname]
    else:
        cloud_factor_dict[pname] = 0.5

# ==========================
# 5) OTHER SIMULATION SETTINGS
# ==========================
AREA_SIZE_KM = 10.0
IMAGE_PIXELS = 500
AXIS_MARGIN = 50
SOLAR_PANEL_SIZE = 20

NUM_CLOUDS = 10
CLOUD_SPEED_PIXELS_PER_FRAME = 2.0
CLOUD_DIRECTION_DEG = 0  # Clouds move rightward

BACKGROUND_COLOR = (102, 204, 102)
AXIS_COLOR = (0, 0, 0)
PANEL_COLOR = (50, 50, 150)
PANEL_LABEL_COLOR = (255, 255, 255)
CLOUD_COLOR = (200, 200, 200, 180)
AXIS_LABEL_FONT_SIZE = 8
PANEL_LABEL_FONT_SIZE = 12

# ==========================
# 6) HELPER FUNCTIONS
# ==========================
def coverage_ratio(panel_box, cloud_box):
    """Compute fraction (0 to 1) of panel_box overlapped by cloud_box."""
    px1, py1, px2, py2 = panel_box
    cx1, cy1, cx2, cy2 = cloud_box
    overlap_x = max(0, min(px2, cx2) - max(px1, cx1))
    overlap_y = max(0, min(py2, cy2) - max(py1, cy1))
    overlap_area = overlap_x * overlap_y
    panel_area = (px2 - px1) * (py2 - py1)
    return overlap_area / panel_area if panel_area > 0 else 0

def create_clouds():
    """Generate a list of cloud dictionaries with random sizes and positions."""
    direction_rad = math.radians(CLOUD_DIRECTION_DEG)
    dx = CLOUD_SPEED_PIXELS_PER_FRAME * math.cos(direction_rad)
    dy = CLOUD_SPEED_PIXELS_PER_FRAME * math.sin(direction_rad)
    clouds = []
    for _ in range(NUM_CLOUDS):
        w = int(np.random.rand() * 80 + 20)
        h = int(np.random.rand() * 50 + 20)
        x_init = int(np.random.rand() * 100) - w
        y_init = np.random.randint(0, IMAGE_PIXELS - h)
        clouds.append({
            'x': x_init,
            'y': y_init,
            'width': w,
            'height': h,
            'dx': dx,
            'dy': dy
        })
    return clouds

def create_ground_image(panels):
    """Create a static ground image with coordinate axes and markers for all panels."""
    ground_img = Image.new("RGB", (IMAGE_PIXELS, IMAGE_PIXELS), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(ground_img)
    
    try:
        axis_font = ImageFont.truetype("arial.ttf", AXIS_LABEL_FONT_SIZE)
        panel_font = ImageFont.truetype("arial.ttf", PANEL_LABEL_FONT_SIZE)
    except IOError:
        axis_font = ImageFont.load_default()
        panel_font = ImageFont.load_default()
    
    axis_length = IMAGE_PIXELS - 2 * AXIS_MARGIN
    x0 = AXIS_MARGIN
    y0 = AXIS_MARGIN
    x1 = x0 + axis_length
    y1 = y0 + axis_length

    # Draw axes
    draw.line([(x0, y1), (x1, y1)], fill=AXIS_COLOR, width=1)
    draw.line([(x0, y0), (x0, y1)], fill=AXIS_COLOR, width=1)
    
    # Draw X-axis ticks & labels
    for i in range(int(AREA_SIZE_KM) + 1):
        px = x0 + int((i / AREA_SIZE_KM) * axis_length)
        draw.line([(px, y1), (px, y1 + 5)], fill=AXIS_COLOR, width=1)
        draw.text((px, y1 + 7), f"{i}", fill=AXIS_COLOR, font=axis_font, anchor="mt")
    
    # Draw Y-axis ticks & labels
    for i in range(int(AREA_SIZE_KM) + 1):
        py = y1 - int((i / AREA_SIZE_KM) * axis_length)
        draw.line([(x0, py), (x0 - 5, py)], fill=AXIS_COLOR, width=1)
        draw.text((x0 - 7, py), f"{i}", fill=AXIS_COLOR, font=axis_font, anchor="rm")
    
    # Draw each panel and label it.
    for panel in panels:
        x_km, y_km = panel['x_km'], panel['y_km']
        px = x0 + int((x_km / AREA_SIZE_KM) * axis_length)
        py = y1 - int((y_km / AREA_SIZE_KM) * axis_length)
        box = [(px - SOLAR_PANEL_SIZE // 2, py - SOLAR_PANEL_SIZE // 2),
               (px + SOLAR_PANEL_SIZE // 2, py + SOLAR_PANEL_SIZE // 2)]
        draw.rectangle(box, fill=PANEL_COLOR)
        draw.text((px, py - SOLAR_PANEL_SIZE // 2 - 5), panel['name'],
                  fill=PANEL_LABEL_COLOR, font=panel_font, anchor="mb")
    
    return ground_img

def get_time_string(frame_index):
    """Return a time string (HH:MM) for the given frame (5-min intervals starting at 06:20)."""
    minutes = 6 * 60 + 20 + frame_index * 5
    h = minutes // 60
    m = minutes % 60
    return f"{int(h):02d}:{int(m):02d}"

def compose_frame(base_ground, clouds, frame_index, panels):
    """
    For each frame:
      - For each panel, retrieve its smooth baseline generation using its panel-specific spline.
      - Compute the cloud coverage over that panel.
      - Compute final generation = baseline * (1 - coverage * panel_cloud_factor).
      - Overlay current time, total generation, average generation, and print detailed cloud effect info.
    Returns the image array and a dictionary with detailed info per panel.
    """
    from PIL import ImageDraw
    
    # Copy the base ground image.
    frame_img = base_ground.copy()
    
    # Update and draw clouds.
    cloud_layer = Image.new("RGBA", (IMAGE_PIXELS, IMAGE_PIXELS), (0, 0, 0, 0))
    draw_clouds = ImageDraw.Draw(cloud_layer, "RGBA")
    for c in clouds:
        c['x'] += c['dx']
        c['y'] += c['dy']
        if c['x'] > IMAGE_PIXELS + c['width']:
            c['x'] = -c['width']
            c['y'] = np.random.randint(0, IMAGE_PIXELS - c['height'])
        c_box = (c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
        draw_clouds.ellipse(c_box, fill=CLOUD_COLOR)
    frame_img.paste(cloud_layer, (0, 0), mask=cloud_layer)
    
    # For each panel, compute its final generation and collect details.
    axis_length = IMAGE_PIXELS - 2 * AXIS_MARGIN
    x0 = AXIS_MARGIN
    y1 = AXIS_MARGIN + axis_length
    panel_info = {}
    
    for panel in panels:
        pname = panel['name']
        baseline = spline_dict[pname](frame_index)
        px = x0 + int((panel['x_km'] / AREA_SIZE_KM) * axis_length)
        py = y1 - int((panel['y_km'] / AREA_SIZE_KM) * axis_length)
        box = (px - SOLAR_PANEL_SIZE // 2, py - SOLAR_PANEL_SIZE // 2,
               px + SOLAR_PANEL_SIZE // 2, py + SOLAR_PANEL_SIZE // 2)
        max_cov = 0.0
        for c in clouds:
            c_box = (c['x'], c['y'], c['x'] + c['width'], c['y'] + c['height'])
            cov = coverage_ratio(box, c_box)
            max_cov = max(max_cov, cov)
        cf = cloud_factor_dict.get(pname, 0.5)
        effective_reduction = max_cov * cf
        final_gen = baseline * (1 - effective_reduction)
        panel_info[pname] = {
            "baseline": baseline,
            "coverage": max_cov,
            "cloud_factor": cf,
            "effective_reduction": effective_reduction,
            "final_gen": final_gen
        }
    
    total_gen = sum(info["final_gen"] for info in panel_info.values())
    avg_gen = total_gen / len(panel_info)
    avg_reduction = sum(info["effective_reduction"] for info in panel_info.values()) / len(panel_info)
    current_time_str = get_time_string(frame_index)
    
    draw_overlay = ImageDraw.Draw(frame_img)
    overlay_text = (f"Time: {current_time_str}\n"
                    f"Total Gen: {total_gen:.1f} kW\n"
                    f"Avg Gen: {avg_gen:.1f} kW\n"
                    f"Avg Reduction: {avg_reduction*100:.0f}%")
    draw_overlay.text((10, 10), overlay_text, fill=(255, 0, 0))
    
    return np.array(frame_img), panel_info

# ==========================
# 7) MAIN SCRIPT
# ==========================
def main():
    global spline_dict
    spline_dict = {}
    global cloud_factor_dict
    baseline_data_dict = {}
    for pname in panel_names:
        if pname == "A":
            baseline_data_dict[pname] = custom_baseline_A
        elif pname == "D":
            baseline_data_dict[pname] = custom_baseline_D
        elif pname == "K":
            baseline_data_dict[pname] = custom_baseline_K
        elif pname == "J":
            baseline_data_dict[pname] = custom_baseline_J
        elif pname == "L":
            baseline_data_dict[pname] = custom_baseline_L
        elif pname == "G":
            baseline_data_dict[pname] = custom_baseline_G
        else:
            baseline_data_dict[pname] = baseline_data_common
        # If the array's length is not TOTAL_POINTS, resample it.
        if len(baseline_data_dict[pname]) != TOTAL_POINTS:
            x_old = np.linspace(0, len(baseline_data_dict[pname]) - 1, len(baseline_data_dict[pname]))
            x_new = np.linspace(0, len(baseline_data_dict[pname]) - 1, TOTAL_POINTS)
            baseline_data_dict[pname] = np.interp(x_new, x_old, baseline_data_dict[pname])
        spline_dict[pname] = CubicSpline(np.linspace(0, TOTAL_POINTS - 1, TOTAL_POINTS),
                                         baseline_data_dict[pname])
    
    base_ground = create_ground_image(panels)
    clouds = create_clouds()
    
    fig, ax = plt.subplots()
    plt.axis("off")
    im = ax.imshow(np.array(base_ground), animated=True)
    
    def update(frame_index):
        frame_arr, panel_info = compose_frame(base_ground, clouds, frame_index, panels)
        im.set_data(frame_arr)
        print(f"\nTime {get_time_string(frame_index)} - Panel Details:")
        for pname in sorted(panel_info.keys()):
            info = panel_info[pname]
            print(f"  {pname}: Baseline = {info['baseline']:.2f} kW, Coverage = {info['coverage']:.2f}, "
                  f"Cloud Factor = {info['cloud_factor']:.2f}, Effective Reduction = {info['effective_reduction']*100:.0f}%, "
                  f"Final Gen = {info['final_gen']:.2f} kW")
        return [im]
    
    ani = FuncAnimation(fig, update, frames=range(TOTAL_FRAMES),
                        interval=INTERVAL_MS, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
