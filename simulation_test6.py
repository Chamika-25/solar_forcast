import math
import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

##############################
# 1. GLOBAL CONSTANTS & SETUP
##############################
TOTAL_FRAMES = 288          # 24 hours at 5-minute intervals
INTERVAL_MS = 50            # 20 FPS (50ms per frame)
FRAMES_PER_HOUR = 12        # 12 frames per hour
AREA_SIZE_KM = 10.0         # 10km x 10km area
IMAGE_PIXELS = 800          # Increased resolution for smooth movement
AXIS_MARGIN = 50
SOLAR_PANEL_SIZE = 25
NUM_CLOUDS = 8              # Reduced number for better visibility
COVERAGE_THRESHOLD = 0.05   # Lower threshold for gradual changes

# Visual parameters
BACKGROUND_COLOR = (102, 204, 102)
PANEL_COLOR = (50, 50, 150)
CLOUD_BASE_COLOR = (200, 200, 200, 180)
CLOUD_OPACITY_RAMP = 0.15   # Opacity change per frame
CLOUD_SPEED_SCALE = 0.4     # Reduced speed for smoother movement

# Solar panel configuration
panel_names = list("ABCDEFGHIJKLMNOPQRST")
default_coords = [
    (8.3, 4.2), (1.6, 9.8), (6.7, 5.1), (9.2, 0.8), (4.5, 7.1),
    (7.3, 8.6), (0.6, 6.3), (3.9, 2.0), (5.8, 8.4), (3.1, 4.9),
    (8.2, 9.9), (5.7, 5.6), (2.9, 0.3), (6.4, 7.8), (9.3, 3.9),
    (5.1, 6.9), (7.0, 9.2), (1.2, 4.6), (9.5, 2.5), (5.1, 1.2),
]
custom_coords = {
    "A": (8.3, 4.2), "D": (9.2, 0.8), "G": (0.6, 6.3),
    "J": (3.9, 2.0), "K": (0.6, 6.3), "L": (3.1, 4.9)
}
panels = []
for i, pname in enumerate(panel_names):
    if i >= len(default_coords):
        break
    coord = custom_coords.get(pname, default_coords[i])
    panels.append({'name': pname, 'x_km': coord[0], 'y_km': coord[1]})

# Cloud factors
custom_cloud_factors = {"A": 0.7, "D": 0.65, "K": 0.55, "J": 0.6, "L": 0.6, "G": 0.55}
cloud_factor_dict = {pname: custom_cloud_factors.get(pname, 0.5) for pname in panel_names}

# Baseline generation data
time_idx = np.linspace(0, TOTAL_FRAMES - 1, TOTAL_FRAMES)
baseline_data_dict = {}
for pname in panel_names:
    arr = np.array([0.1, 0.2, 0.3])  # Simplified for example
    if len(arr) != TOTAL_FRAMES:
        x_old = np.linspace(0, len(arr)-1, len(arr))
        x_new = np.linspace(0, len(arr)-1, TOTAL_FRAMES)
        arr = np.interp(x_new, x_old, arr)
    baseline_data_dict[pname] = CubicSpline(time_idx, arr)

##############################
# 2. ENHANCED CLOUD CLASS
##############################
class Cloud:
    def __init__(self, birth_frame):
        self.width = int(np.random.uniform(100, 160))
        self.height = int(np.random.uniform(60, 100))
        self.x = np.random.uniform(-self.width, IMAGE_PIXELS)
        self.y = np.random.uniform(-self.height, IMAGE_PIXELS)
        self.opacity = 0.0
        self.active = False
        self.birth_frame = birth_frame
        self.lifetime = 0
        
    def update(self, dx, dy, frame_idx):
        # Smooth position update with boundary wrapping
        self.x = (self.x + dx) % (IMAGE_PIXELS + self.width*2)
        self.y = (self.y + dy) % (IMAGE_PIXELS + self.height*2)
        
        # Manage cloud lifecycle
        self.lifetime = frame_idx - self.birth_frame
        screen_margin = 200
        self.active = (
            -screen_margin < self.x < IMAGE_PIXELS + screen_margin and 
            -screen_margin < self.y < IMAGE_PIXELS + screen_margin
        )
        
        # Smooth opacity transitions
        target_opacity = 180 if self.active else 0
        self.opacity += (target_opacity - self.opacity) * CLOUD_OPACITY_RAMP
        self.opacity = np.clip(self.opacity, 0, 180)

##############################
# 3. WEATHER DATA & CLOUD SYSTEM
##############################
class WeatherSystem:
    def __init__(self, date_str):
        self.cc_hourly, self.wspd_hourly, self.wdir_hourly = self.get_weather_data(date_str)
        self.cc_5min = self.upsample_to_5min(self.cc_hourly, date_str)[:TOTAL_FRAMES]
        self.wspd_5min = self.upsample_to_5min(self.wspd_hourly, date_str)[:TOTAL_FRAMES]
        self.wdir_5min = self.upsample_to_5min(self.wdir_hourly, date_str)[:TOTAL_FRAMES]
        self.clouds = []
        self.current_cloud_id = 0
        
    def get_weather_data(self, date_str):
        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 35.69,
            "longitude": 139.69,
            "hourly": "cloudcover,windspeed_10m,winddirection_10m",
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "auto"
        }
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            data = resp.json()
            return (
                data["hourly"]["cloudcover"],
                data["hourly"]["windspeed_10m"],
                data["hourly"]["winddirection_10m"]
            )
        except Exception as e:
            print("API error:", e)
            return [50.0]*24, [5.0]*24, [270.0]*24
    
    def upsample_to_5min(self, arr_hourly, date_str):
        idx_hourly = pd.date_range(start=date_str, periods=len(arr_hourly), freq="H")
        idx_5min = pd.date_range(start=date_str, periods=len(arr_hourly)*12, freq="5min")
        return pd.Series(arr_hourly, index=idx_hourly).reindex(idx_5min).interpolate(method="time").values
    
    def update_clouds(self, frame_idx):
        if len(self.clouds) < NUM_CLOUDS and frame_idx % 15 == 0:
            self.clouds.append(Cloud(frame_idx))
            
        # Remove expired clouds
        self.clouds = [c for c in self.clouds if c.lifetime < 3600]
        
        # Calculate interpolated wind parameters
        prev_idx = max(0, frame_idx-1)
        alpha = frame_idx % 1
        
        ws = self.wspd_5min[prev_idx] * (1-alpha) + self.wspd_5min[frame_idx] * alpha
        wd = self.wdir_5min[prev_idx] * (1-alpha) + self.wdir_5min[frame_idx] * alpha
        
        # Calculate movement vectors
        wd_rad = math.radians(wd)
        dist_km = ws * 0.06 * 5 * CLOUD_SPEED_SCALE
        px_per_km = IMAGE_PIXELS / AREA_SIZE_KM
        dx = dist_km * math.cos(wd_rad) * px_per_km
        dy = -dist_km * math.sin(wd_rad) * px_per_km
        
        # Update all clouds
        for cloud in self.clouds:
            cloud.update(dx, dy, frame_idx)

##############################
# 4. COVERAGE CALCULATION
##############################
def calculate_coverage(panel_pos, clouds, cc_percent):
    coverage = 0.0
    panel_x, panel_y = panel_pos
    
    for cloud in clouds:
        if not cloud.active or cloud.opacity < 10:
            continue
            
        cloud_center_x = cloud.x + cloud.width/2
        cloud_center_y = cloud.y + cloud.height/2
        distance_x = abs(panel_x - cloud_center_x)
        distance_y = abs(panel_y - cloud_center_y)
        
        max_dist = max(cloud.width, cloud.height) * 0.7
        distance = math.hypot(distance_x, distance_y)
        coverage += np.clip(1 - distance/max_dist, 0, 1) * (cloud.opacity/180)
    
    return np.clip(coverage * (cc_percent/100), 0, 1)

##############################
# 5. VISUALIZATION SYSTEM
##############################
class VisualizationSystem:
    def __init__(self):
        self.base_map = self.create_base_map()
        self.font = self.load_font()
        
    def create_base_map(self):
        base = Image.new("RGB", (IMAGE_PIXELS, IMAGE_PIXELS), BACKGROUND_COLOR)
        d = ImageDraw.Draw(base)
        axis_length = IMAGE_PIXELS - 2 * AXIS_MARGIN
        x0 = AXIS_MARGIN
        y0 = AXIS_MARGIN
        x1 = x0 + axis_length
        y1 = y0 + axis_length

        # Draw grid and labels
        d.line([(x0, y1), (x1, y1)], fill=(0,0,0), width=1)
        d.line([(x0, y0), (x0, y1)], fill=(0,0,0), width=1)
        
        for i in range(int(AREA_SIZE_KM)+1):
            px = x0 + int((i/AREA_SIZE_KM)*axis_length)
            d.line([(px, y1), (px, y1+5)], fill=(0,0,0))
            d.text((px, y1+7), f"{i}", fill=(0,0,0))
            
        for i in range(int(AREA_SIZE_KM)+1):
            py = y1 - int((i/AREA_SIZE_KM)*axis_length)
            d.line([(x0-5, py), (x0, py)], fill=(0,0,0))
            d.text((x0-25, py), f"{i}", fill=(0,0,0))

        # Draw solar panels
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        for p in panels:
            px_km, py_km = p['x_km'], p['y_km']
            px = x0 + int((px_km/AREA_SIZE_KM)*axis_length)
            py = y1 - int((py_km/AREA_SIZE_KM)*axis_length)
            box = (px-SOLAR_PANEL_SIZE//2, py-SOLAR_PANEL_SIZE//2,
                   px+SOLAR_PANEL_SIZE//2, py+SOLAR_PANEL_SIZE//2)
            d.rectangle(box, fill=PANEL_COLOR)
            d.text((px, py-SOLAR_PANEL_SIZE//2-15), p['name'], fill=(255,255,255), font=font)
        
        return base
    
    def load_font(self):
        try:
            return ImageFont.truetype("arial.ttf", 14)
        except:
            return ImageFont.load_default()
    
    def draw_clouds(self, base_img, clouds):
        overlay = Image.new("RGBA", (IMAGE_PIXELS, IMAGE_PIXELS), (0,0,0,0))
        dd = ImageDraw.Draw(overlay)
        
        for cloud in clouds:
            if cloud.opacity < 5:
                continue
                
            base_opacity = int(cloud.opacity * 0.8)
            for i in range(3):
                offset = i * 4
                box = (
                    cloud.x - offset, cloud.y - offset,
                    cloud.x + cloud.width + offset,
                    cloud.y + cloud.height + offset
                )
                dd.ellipse(box, fill=(200, 200, 200, int(base_opacity * (0.7**i))))
        
        base_img.paste(overlay, (0, 0), overlay)
    
    def draw_ui(self, img, frame_idx, total_gen, weather):
        d = ImageDraw.Draw(img)
        timestr = get_time_string(frame_idx)
        
        # Main info box
        text = (
            f"Time: {timestr}\n"
            f"Cloud Cover: {weather['cc']:.0f}%\n"
            f"Wind: {weather['ws']:.1f}m/s @ {weather['wd']:.0f}°\n"
            f"Total Generation: {total_gen:.1f} kW"
        )
        d.rectangle([10, 10, 220, 130], fill=(0,0,0,128))
        d.text((15, 15), text, fill=(255,255,0), font=self.font)

##############################
# 6. MAIN SIMULATION
##############################
def get_time_string(frame_idx):
    minutes = 6*60 + 20 + frame_idx*5
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"

def main():
    date_str = "2025-04-10"
    weather_system = WeatherSystem(date_str)
    viz_system = VisualizationSystem()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.axis("off")
    im = ax.imshow(np.array(viz_system.base_map), animated=True)
    
    def update(frame_idx):
        if frame_idx >= TOTAL_FRAMES:
            frame_idx = TOTAL_FRAMES - 1
        
        weather_system.update_clouds(frame_idx)
        frame_img = viz_system.base_map.copy()
        viz_system.draw_clouds(frame_img, weather_system.clouds)
        
        # Calculate generation
        total_gen = 0
        panel_generation = {}
        axis_length = IMAGE_PIXELS - 2 * AXIS_MARGIN
        x0 = AXIS_MARGIN
        y1_px = x0 + axis_length
        
        for p in panels:
            pname = p['name']
            baseline_val = baseline_data_dict[pname](frame_idx)
            px = x0 + int((p['x_km'] / AREA_SIZE_KM) * axis_length)
            py = y1_px - int((p['y_km'] / AREA_SIZE_KM) * axis_length)
            panel_pos = (px + SOLAR_PANEL_SIZE//2, py + SOLAR_PANEL_SIZE//2)
            
            coverage = calculate_coverage(
                panel_pos, 
                weather_system.clouds,
                weather_system.cc_5min[frame_idx]
            )
            eff_reduction = coverage * cloud_factor_dict[pname]
            final_gen = baseline_val * (1 - eff_reduction)
            
            panel_generation[pname] = final_gen
            total_gen += final_gen
        
        # Draw UI
        weather_data = {
            'cc': weather_system.cc_5min[frame_idx],
            'ws': weather_system.wspd_5min[frame_idx],
            'wd': weather_system.wdir_5min[frame_idx]
        }
        viz_system.draw_ui(frame_img, frame_idx, total_gen, weather_data)
        
        im.set_data(np.array(frame_img))
        return [im]
    
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=INTERVAL_MS, blit=False)
    plt.show()

if __name__ == "__main__":
    main()