from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
area_size_km = 10.0  # Area size in km
image_pixels = 500
num_solar_panels = 50
num_clouds = 20
solar_panel_size = 8  # Slightly smaller panel size for labels to fit better
panel_label_font_size = 6
axis_label_font_size = 8
axis_color = (0, 0, 0)  # Black for axes and labels
panel_label_color = (255, 255, 255) # White for panel labels

# --- Coordinate System Configuration ---
x_axis_start_pixel = 50
y_axis_start_pixel = 50
axis_length_pixels = image_pixels - 100 # Leave margin on edges
x_axis_end_pixel = x_axis_start_pixel + axis_length_pixels
y_axis_end_pixel = y_axis_start_pixel + axis_length_pixels
x_km_start = 0.0
x_km_end = area_size_km
y_km_start = 0.0
y_km_end = area_size_km
x_ticks_km = 1.0  # Tick interval in km for X-axis
y_ticks_km = 1.0  # Tick interval in km for Y-axis

# --- Step 1: Create Background with Coordinate Axes ---
background_color = (102, 204, 102)
background_image = Image.new("RGB", (image_pixels, image_pixels), background_color)
draw = ImageDraw.Draw(background_image)

# Load fonts
try:
    axis_label_font = ImageFont.truetype("arial.ttf", axis_label_font_size)
    panel_label_font = ImageFont.truetype("arial.ttf", panel_label_font_size)
except IOError:
    axis_label_font = ImageFont.load_default()
    panel_label_font = ImageFont.load_default()

# Draw X and Y axes
draw.line([(x_axis_start_pixel, y_axis_end_pixel), (x_axis_end_pixel, y_axis_end_pixel)], fill=axis_color, width=1) # X-axis
draw.line([(x_axis_start_pixel, y_axis_start_pixel), (x_axis_start_pixel, y_axis_end_pixel)], fill=axis_color, width=1) # Y-axis

# Add X-axis tick marks and labels
num_x_ticks = int((x_km_end - x_km_start) / x_ticks_km) + 1
for i in range(num_x_ticks):
    km_value = x_km_start + i * x_ticks_km
    pixel_x = x_axis_start_pixel + int((km_value / area_size_km) * axis_length_pixels)
    draw.line([(pixel_x, y_axis_end_pixel), (pixel_x, y_axis_end_pixel + 5)], fill=axis_color, width=1) # Tick mark
    draw.text((pixel_x, y_axis_end_pixel + 7), f"{km_value:.0f}", fill=axis_color, font=axis_label_font, anchor="mt") # Label

# Add Y-axis tick marks and labels
num_y_ticks = int((y_km_end - y_km_start) / y_ticks_km) + 1
for i in range(num_y_ticks):
    km_value = y_km_start + i * y_ticks_km
    pixel_y = y_axis_end_pixel - int((km_value / area_size_km) * axis_length_pixels) # Y-axis is inverted in pixels
    draw.line([(x_axis_start_pixel, pixel_y), (x_axis_start_pixel - 5, pixel_y)], fill=axis_color, width=1) # Tick mark
    draw.text((x_axis_start_pixel - 7, pixel_y), f"{km_value:.0f}", fill=axis_color, font=axis_label_font, anchor="rm") # Label

# --- Step 2: Place Solar Panels with X,Y Coordinate Labels ---
solar_panel_color = (50, 50, 150)
panel_locations_km = [] # Store locations in km
panel_ids = {}

for i in range(num_solar_panels):
    # Generate random location in km
    x_km = np.random.rand() * area_size_km
    y_km = np.random.rand() * area_size_km
    panel_locations_km.append((x_km, y_km))

    # Convert km coordinates to pixel coordinates
    x_pixel = x_axis_start_pixel + int((x_km / area_size_km) * axis_length_pixels)
    y_pixel = y_axis_end_pixel - int((y_km / area_size_km) * axis_length_pixels) # Invert Y for pixel coords

    panel_ids[f"P{i+1}"] = (x_km, y_km) # Store ID and km location

    # Draw solar panel
    panel_box = [(x_pixel - solar_panel_size // 2, y_pixel - solar_panel_size // 2),
                 (x_pixel + solar_panel_size // 2, y_pixel + solar_panel_size // 2)]
    draw.rectangle(panel_box, fill=solar_panel_color)

    # Draw coordinate label (X,Y in km) above the panel
    label_x = x_pixel
    label_y = y_pixel - solar_panel_size // 2 - 5  # Position label slightly above panel
    label_text = f"({x_km:.1f},{y_km:.1f})" # Format coordinates to 1 decimal place
    draw.text((label_x, label_y), label_text, fill=panel_label_color, font=panel_label_font, anchor="mb") # "mb" anchor: middle-bottom


# --- Step 3 & 4: Generate & Place Clouds (Simplified Clouds - Ellipses) ---
cloud_color = (200, 200, 200, 180)

for _ in range(num_clouds):
    cloud_width = int(np.random.rand() * 100 + 10)
    cloud_height = int(np.random.rand() * 50 + 30)
    cloud_x = int(np.random.rand() * image_pixels - cloud_width // 2)
    cloud_y = int(np.random.rand() * image_pixels - cloud_height // 2)

    cloud_box = [(cloud_x, cloud_y), (cloud_x + cloud_width, cloud_y + cloud_height)]
    draw.ellipse(cloud_box, fill=cloud_color)


# --- Step 5: Visualization ---
plt.imshow(background_image)
plt.title("Solar Panel Simulation with X,Y Coordinates (km)")
plt.axis('off')
plt.show()

# --- Optional: Save the image ---
# background_image.save("solar_simulation_xy_coords.png")

# Print Panel Locations and IDs
for panel_id, location_km in panel_ids.items():
    print(f"Panel ID: {panel_id}, Location (km): X={location_km[0]:.2f}, Y={location_km[1]:.2f}")