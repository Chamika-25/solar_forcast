import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
AREA_SIZE_KM = 10.0                 # Ground area side (km)
IMAGE_PIXELS = 500                  # Image size in pixels
AXIS_MARGIN = 50                    # Margin for axes
SOLAR_PANEL_SIZE = 20               # Solar panel drawn as a square (in pixels)

# Use the provided coordinates for solar panel locations (20 panels)
manual_panel_locations_km = [
    (8.3, 4.2),
    (1.6, 9.8),
    (6.7, 5.1),
    (9.2, 0.8),
    (4.5, 7.1),
    (7.3, 8.6),
    (0.6, 6.3),
    (3.9, 2.0),
    (5.8, 8.4),
    (3.1, 4.9),
    (8.2, 9.9),
    (5.7, 5.6),
    (2.9, 0.3),
    (6.4, 7.8),
    (9.3, 3.9),
    (5.1, 6.9),
    (7.0, 9.2),
    (1.2, 4.6),
    (9.5, 2.5),
    (5.1, 1.2)
]

NUM_PANELS = len(manual_panel_locations_km)  # Number of panels (20)

NUM_CLOUDS = 10                     # Number of clouds
CLOUD_SPEED_PIXELS_PER_FRAME = 1.0  # Cloud horizontal speed
CLOUD_DIRECTION_DEG = 0             # 0° => move clouds to the right
TOTAL_FRAMES = 200                  # Total frames for animation
INTERVAL_MS = 50                    # Delay between frames in ms (~20 fps)

# Axis tick spacing in km
X_TICKS_KM = 1.0
Y_TICKS_KM = 1.0

# Colors
BACKGROUND_COLOR = (102, 204, 102)     # greenish background
AXIS_COLOR = (0, 0, 0)                 # black
PANEL_COLOR = (50, 50, 150)            # dark blue for panels
PANEL_LABEL_COLOR = (255, 255, 255)    # white for labels
CLOUD_COLOR = (200, 200, 200, 180)     # semi-transparent gray

# Fonts
AXIS_LABEL_FONT_SIZE = 8
PANEL_LABEL_FONT_SIZE = 6

# ---------------------------------------------------------------------
# 1) CREATE THE STATIC GROUND IMAGE (with panels)
# ---------------------------------------------------------------------
def create_ground_image():
    # Base image for the ground
    ground_img = Image.new("RGB", (IMAGE_PIXELS, IMAGE_PIXELS), BACKGROUND_COLOR)
    draw_ground = ImageDraw.Draw(ground_img)

    # Attempt to load fonts, else use defaults
    try:
        axis_label_font = ImageFont.truetype("arial.ttf", AXIS_LABEL_FONT_SIZE)
        panel_label_font = ImageFont.truetype("arial.ttf", PANEL_LABEL_FONT_SIZE)
    except IOError:
        axis_label_font = ImageFont.load_default()
        panel_label_font = ImageFont.load_default()

    # Compute pixel ranges for axes
    axis_length_pixels = IMAGE_PIXELS - 2 * AXIS_MARGIN
    x_axis_start = AXIS_MARGIN
    y_axis_start = AXIS_MARGIN
    x_axis_end   = x_axis_start + axis_length_pixels
    y_axis_end   = y_axis_start + axis_length_pixels

    # Draw X-axis line
    draw_ground.line([(x_axis_start, y_axis_end), (x_axis_end, y_axis_end)],
                     fill=AXIS_COLOR, width=1)
    # Draw Y-axis line
    draw_ground.line([(x_axis_start, y_axis_start), (x_axis_start, y_axis_end)],
                     fill=AXIS_COLOR, width=1)

    # Draw X-axis tick marks & labels
    num_x_ticks = int(AREA_SIZE_KM / X_TICKS_KM) + 1
    for i in range(num_x_ticks):
        km_val = i * X_TICKS_KM
        px = x_axis_start + int((km_val / AREA_SIZE_KM) * axis_length_pixels)
        draw_ground.line([(px, y_axis_end), (px, y_axis_end + 5)], fill=AXIS_COLOR, width=1)
        draw_ground.text((px, y_axis_end + 7), f"{km_val:.0f}", fill=AXIS_COLOR,
                         font=axis_label_font, anchor="mt")

    # Draw Y-axis tick marks & labels
    num_y_ticks = int(AREA_SIZE_KM / Y_TICKS_KM) + 1
    for i in range(num_y_ticks):
        km_val = i * Y_TICKS_KM
        py = y_axis_end - int((km_val / AREA_SIZE_KM) * axis_length_pixels)
        draw_ground.line([(x_axis_start, py), (x_axis_start - 5, py)], fill=AXIS_COLOR, width=1)
        draw_ground.text((x_axis_start - 7, py), f"{km_val:.0f}", fill=AXIS_COLOR,
                         font=axis_label_font, anchor="rm")

    # Draw solar panels at provided coordinates
    for (x_km, y_km) in manual_panel_locations_km:
        px = x_axis_start + int((x_km / AREA_SIZE_KM) * axis_length_pixels)
        py = y_axis_end - int((y_km / AREA_SIZE_KM) * axis_length_pixels)
        panel_box = [
            (px - SOLAR_PANEL_SIZE // 2, py - SOLAR_PANEL_SIZE // 2),
            (px + SOLAR_PANEL_SIZE // 2, py + SOLAR_PANEL_SIZE // 2)
        ]
        draw_ground.rectangle(panel_box, fill=PANEL_COLOR)
        label_text = f"({x_km:.1f},{y_km:.1f})"
        label_x = px
        label_y = py - SOLAR_PANEL_SIZE // 2 - 5
        draw_ground.text((label_x, label_y), label_text, fill=PANEL_LABEL_COLOR,
                         font=panel_label_font, anchor="mb")
    return ground_img

# ---------------------------------------------------------------------
# 2) CREATE CLOUD LIST
# ---------------------------------------------------------------------
def create_clouds():
    direction_rad = math.radians(CLOUD_DIRECTION_DEG)
    dx = CLOUD_SPEED_PIXELS_PER_FRAME * math.cos(direction_rad)
    dy = CLOUD_SPEED_PIXELS_PER_FRAME * math.sin(direction_rad)

    clist = []
    for _ in range(NUM_CLOUDS):
        w = int(np.random.rand() * 80 + 20)   # random width
        h = int(np.random.rand() * 50 + 20)   # random height
        # Start near the left edge
        x_init = int(np.random.rand() * 100) - w
        y_init = np.random.randint(0, IMAGE_PIXELS - h)
        clist.append({
            'x': x_init,
            'y': y_init,
            'width': w,
            'height': h,
            'dx': dx,
            'dy': dy
        })
    return clist

# ---------------------------------------------------------------------
# 3) COMPOSE A FRAME (GROUND + MOVED CLOUDS)
# ---------------------------------------------------------------------
def compose_frame(base_ground, cloud_list):
    """
    Returns a NumPy array of the final composite frame:
      1) Copies the base ground image.
      2) Moves & draws each cloud on a transparent layer.
      3) Pastes the cloud layer over the ground.
      4) Converts to a NumPy array for imshow.
    """
    frame_img = base_ground.copy()  # Copy of the static ground image
    cloud_layer = Image.new("RGBA", (IMAGE_PIXELS, IMAGE_PIXELS), (0, 0, 0, 0))
    draw_clouds = ImageDraw.Draw(cloud_layer, "RGBA")

    # Move & draw each cloud
    for c in cloud_list:
        c['x'] += c['dx']
        c['y'] += c['dy']
        # Wrap around if the cloud moves off the right edge
        if c['x'] > IMAGE_PIXELS + c['width']:
            c['x'] = -c['width']
            c['y'] = np.random.randint(0, IMAGE_PIXELS - c['height'])
        # Draw an elliptical cloud
        box = [(c['x'], c['y']), (c['x'] + c['width'], c['y'] + c['height'])]
        draw_clouds.ellipse(box, fill=CLOUD_COLOR)

    # Paste the cloud layer over the ground image
    frame_img.paste(cloud_layer, (0, 0), mask=cloud_layer)
    return np.array(frame_img)

# ---------------------------------------------------------------------
# 4) MAIN SCRIPT: CREATE GROUND, CLOUDS, THEN ANIMATE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    base_ground_image = create_ground_image()
    clouds = create_clouds()

    fig, ax = plt.subplots()
    plt.axis('off')
    im = ax.imshow(np.array(base_ground_image), animated=True)

    def update(frame):
        arr = compose_frame(base_ground_image, clouds)
        im.set_data(arr)
        return [im]

    ani = FuncAnimation(fig, update, frames=range(TOTAL_FRAMES),
                        interval=INTERVAL_MS, blit=True)
    plt.show()
