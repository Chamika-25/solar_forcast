import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
import random

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
MAX_CLOUDS = 25             # Maximum number of clouds in the simulation
COVERAGE_THRESHOLD = 0.05   # Lower threshold for gradual changes

# Cloud pattern parameters
CLOUD_PATTERNS = {
    "SCATTERED": {
        "probability": 0.5,       # 50% chance of scattered pattern
        "count_range": (3, 8),    # Few scattered clouds
        "grouping_factor": 0.2,   # Low grouping (more spread out)
        "duration": (30, 90)      # How long this pattern lasts (in frames)
    },
    "BUNCHED": {
        "probability": 0.3,       # 30% chance of bunched pattern
        "count_range": (10, 20),  # Many clouds together
        "grouping_factor": 0.8,   # High grouping (clouds appear in clusters)
        "duration": (15, 45)      # Shorter duration for dense clouds
    },
    "ISOLATED": {
        "probability": 0.2,       # 20% chance of isolated pattern
        "count_range": (1, 3),    # Very few isolated clouds
        "grouping_factor": 0.1,   # Very low grouping (isolated clouds)
        "duration": (20, 60)      # Medium duration for this pattern
    }
}

# Visual parameters
BACKGROUND_COLOR = (102, 204, 102)
PANEL_COLOR = (50, 50, 150)
CLOUD_BASE_COLOR = (255, 255, 255, 180)  # Whiter for cartoon clouds
CLOUD_OPACITY_RAMP = 0.15   # Opacity change per frame
CLOUD_SPEED_SCALE = 0.4     # Reduced speed for smoother movement

# Cloud size parameters - SMALLER CLOUDS
CLOUD_WIDTH_MIN = 40        # Minimum cloud width
CLOUD_WIDTH_MAX = 80        # Maximum cloud width  
CLOUD_HEIGHT_MIN = 25       # Minimum cloud height
CLOUD_HEIGHT_MAX = 50       # Maximum cloud height
CLOUD_SCALE_FACTOR = 0.5    # Overall scaling factor for cloud rendering

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

# Baseline generation data - create a simple day curve
time_idx = np.linspace(0, TOTAL_FRAMES - 1, TOTAL_FRAMES)
baseline_data_dict = {}
for pname in panel_names:
    # Create a simple diurnal curve with peak at noon
    hours = np.linspace(0, 24, TOTAL_FRAMES)
    values = np.sin(np.pi * (hours - 6) / 12) * 0.8  # Peak at noon (hour 12)
    values = np.clip(values, 0, 1)  # Ensure non-negative values
    baseline_data_dict[pname] = CubicSpline(time_idx, values)

##############################
# 2. ENHANCED CLOUD CLASS
##############################
class Cloud:
    def __init__(self, birth_frame, size_factor=1.0, position=None):
        # Allow variable sizes with a size factor
        width_min = int(CLOUD_WIDTH_MIN * size_factor)
        width_max = int(CLOUD_WIDTH_MAX * size_factor)
        height_min = int(CLOUD_HEIGHT_MIN * size_factor)
        height_max = int(CLOUD_HEIGHT_MAX * size_factor)
        
        # Cloud dimensions with randomness
        self.width = int(np.random.uniform(width_min, width_max))
        self.height = int(np.random.uniform(height_min, height_max))
        
        # Position - either random or specified (for cloud clusters)
        if position is None:
            self.x = np.random.uniform(-self.width, IMAGE_PIXELS)
            self.y = np.random.uniform(-self.height, IMAGE_PIXELS)
        else:
            # Add some variation to the position if it's part of a cluster
            variation = 50 * size_factor  # More variation for larger clouds
            self.x = position[0] + np.random.uniform(-variation, variation)
            self.y = position[1] + np.random.uniform(-variation, variation)
        
        self.opacity = 0.0
        self.active = False
        self.birth_frame = birth_frame
        self.lifetime = 0
        
        # Add randomness for cartoon appearance
        self.puff_variation = np.random.uniform(0.8, 1.2, 8)  # Random variations for puffs
        # Add a little rotation for variety
        self.rotation = np.random.uniform(0, 2*np.pi)
        # Random cloud color variation (slight blue or gray tint)
        tint = np.random.randint(0, 20)
        self.color = (255-tint, 255-tint, 255-max(0, tint-10))
        
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
    def __init__(self):
        # Generate synthetic weather data
        # Cloud cover (%)
        self.cc_hourly = self.generate_synthetic_cloud_cover(24)
        # Wind speed (m/s)
        self.wspd_hourly = self.generate_synthetic_wind_speed(24)
        # Wind direction (degrees)
        self.wdir_hourly = self.generate_synthetic_wind_direction(24)
        
        # Upsample to 5-minute intervals
        self.cc_5min = self.upsample_to_5min(self.cc_hourly)
        self.wspd_5min = self.upsample_to_5min(self.wspd_hourly)
        self.wdir_5min = self.upsample_to_5min(self.wdir_hourly)
        
        self.clouds = []
        
        # Cloud pattern control
        self.current_pattern = "SCATTERED"  # Start with scattered pattern
        self.pattern_change_frame = 0
        self.next_pattern_change = random.randint(*CLOUD_PATTERNS["SCATTERED"]["duration"])
        self.target_cloud_count = random.randint(*CLOUD_PATTERNS["SCATTERED"]["count_range"])
        
    def generate_synthetic_cloud_cover(self, hours):
        # Generate realistic cloud cover pattern with some variation
        base = 40 + 20 * np.sin(np.linspace(0, 2*np.pi, hours))
        noise = np.random.normal(0, 10, hours)
        cc = np.clip(base + noise, 0, 100)
        return cc
        
    def generate_synthetic_wind_speed(self, hours):
        # Generate wind speed that increases during the day
        base = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, hours))
        noise = np.random.normal(0, 0.5, hours)
        speed = np.clip(base + noise, 0.5, 10)
        return speed
        
    def generate_synthetic_wind_direction(self, hours):
        # Start with westerly winds (270°) and add gradual rotation
        base = 270 + 45 * np.sin(np.linspace(0, np.pi, hours))
        noise = np.random.normal(0, 15, hours)
        direction = (base + noise) % 360
        return direction
    
    def upsample_to_5min(self, arr_hourly):
        # Create a smooth interpolation between hourly values
        x_hourly = np.arange(len(arr_hourly))
        x_5min = np.linspace(0, len(arr_hourly)-1, len(arr_hourly)*12)
        return np.interp(x_5min, x_hourly, arr_hourly)
    
    def select_new_pattern(self):
        # Choose a new cloud pattern based on probabilities
        r = random.random()
        cum_prob = 0
        for pattern, params in CLOUD_PATTERNS.items():
            cum_prob += params["probability"]
            if r <= cum_prob:
                self.current_pattern = pattern
                break
        
        # Set parameters for this pattern
        pattern_params = CLOUD_PATTERNS[self.current_pattern]
        self.target_cloud_count = random.randint(*pattern_params["count_range"])
        self.next_pattern_change = random.randint(*pattern_params["duration"])
    
    def create_clouds(self, frame_idx):
        # Check if it's time to change the pattern
        if frame_idx - self.pattern_change_frame >= self.next_pattern_change:
            self.pattern_change_frame = frame_idx
            self.select_new_pattern()
            # Print for debugging
            print(f"Frame {frame_idx}: Changing to {self.current_pattern} pattern, target {self.target_cloud_count} clouds")
        
        # Add or remove clouds to match the target count for the current pattern
        current_count = len(self.clouds)
        
        # If we need more clouds, add them according to the pattern
        if current_count < self.target_cloud_count and current_count < MAX_CLOUDS:
            pattern_params = CLOUD_PATTERNS[self.current_pattern]
            grouping_factor = pattern_params["grouping_factor"]
            
            # Determine if this should be a new cluster or an addition to existing clouds
            if random.random() < grouping_factor and current_count > 0:
                # Add cloud near an existing cloud to create a cluster
                parent_cloud = random.choice(self.clouds)
                position = (parent_cloud.x, parent_cloud.y)
                
                # Vary the size within clusters
                size_factor = 0.8 + random.random() * 0.4  # 0.8 to 1.2
                
                # Create new cloud near the parent
                self.clouds.append(Cloud(frame_idx, size_factor, position))
            else:
                # Create a completely new cloud with random position
                size_factor = 0.7 + random.random() * 0.6  # 0.7 to 1.3
                self.clouds.append(Cloud(frame_idx, size_factor))
    
    def update_clouds(self, frame_idx):
        # First, create or update cloud pattern
        self.create_clouds(frame_idx)
            
        # Remove expired clouds or limit to target count
        self.clouds = [c for c in self.clouds if c.lifetime < 3600]
        
        # If we still have too many clouds, remove some of the oldest ones
        while len(self.clouds) > self.target_cloud_count:
            # Find the oldest clouds and remove them
            oldest_idx = 0
            oldest_lifetime = -1
            for i, cloud in enumerate(self.clouds):
                if cloud.lifetime > oldest_lifetime:
                    oldest_lifetime = cloud.lifetime
                    oldest_idx = i
            
            # Remove the oldest cloud
            if oldest_idx < len(self.clouds):
                self.clouds.pop(oldest_idx)
        
        # Calculate current weather parameters
        ws = self.wspd_5min[min(frame_idx, len(self.wspd_5min)-1)]
        wd = self.wdir_5min[min(frame_idx, len(self.wdir_5min)-1)]
        
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
        
        # Adjust coverage calculation for smaller clouds
        max_dist = max(cloud.width, cloud.height) * 0.6  # Reduced from 0.7
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
            d.text((px, py-SOLAR_PANEL_SIZE//2-15), p['name'], fill=(0,0,0), font=font)
        
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
                
            # Calculate base opacity for this cloud
            base_opacity = int(cloud.opacity)
            cloud_color_with_opacity = cloud.color + (base_opacity,)
            
            # Define cloud parameters - SMALLER
            cloud_center_x = cloud.x + cloud.width/2
            cloud_center_y = cloud.y + cloud.height/2
            base_radius = min(cloud.width, cloud.height) * 0.25 * CLOUD_SCALE_FACTOR  # Scale down
            
            # Create a cartoon-style cloud with multiple overlapping circles
            # Main cloud body - a larger central circle
            main_radius = base_radius * 1.2
            dd.ellipse(
                (cloud_center_x - main_radius, cloud_center_y - main_radius,
                 cloud_center_x + main_radius, cloud_center_y + main_radius),
                fill=cloud_color_with_opacity
            )
            
            # Add 5-7 smaller "puff" circles around the main circle
            num_puffs = 5  # Reduced number of puffs for smaller clouds
            for i in range(num_puffs):
                # Calculate position around the main circle, with rotation
                angle = cloud.rotation + (i / num_puffs) * 2 * math.pi
                distance = base_radius * 0.9
                puff_x = cloud_center_x + math.cos(angle) * distance
                puff_y = cloud_center_y + math.sin(angle) * distance
                
                # Vary the puff sizes slightly for a more natural look
                puff_radius = base_radius * (0.6 + 0.3 * (i % 3) / 2) * cloud.puff_variation[i % len(cloud.puff_variation)]
                
                dd.ellipse(
                    (puff_x - puff_radius, puff_y - puff_radius,
                     puff_x + puff_radius, puff_y + puff_radius),
                    fill=cloud_color_with_opacity
                )
            
            # Add smaller detail puffs - fewer for small clouds
            for i in range(3):  # Reduced from 4
                angle = cloud.rotation + ((i + 0.5) / 3) * 2 * math.pi
                distance = base_radius * 1.3  # Slightly reduced
                puff_x = cloud_center_x + math.cos(angle) * distance
                puff_y = cloud_center_y + math.sin(angle) * distance
                puff_radius = base_radius * 0.4 * cloud.puff_variation[i % len(cloud.puff_variation)]
                
                dd.ellipse(
                    (puff_x - puff_radius, puff_y - puff_radius,
                     puff_x + puff_radius, puff_y + puff_radius),
                    fill=cloud_color_with_opacity
                )
        
        # Merge the cloud overlay onto the base image
        base_img.paste(overlay, (0, 0), overlay)
    
    def draw_ui(self, img, frame_idx, total_gen, weather):
        d = ImageDraw.Draw(img)
        timestr = get_time_string(frame_idx)
        
        # Main info box
        text = (
            f"Time: {timestr}\n"
            f"Cloud Cover: {weather['cc']:.0f}%\n"
            f"Wind: {weather['ws']:.1f}m/s @ {weather['wd']:.0f}°\n"
            f"Cloud Pattern: {weather['pattern']}\n"
            f"Total Generation: {total_gen:.1f} kW"
        )
        d.rectangle([10, 10, 220, 150], fill=(0,0,0,128))
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
    weather_system = WeatherSystem()
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
            
            # Get cloud cover at this time
            cc_percent = weather_system.cc_5min[min(frame_idx, len(weather_system.cc_5min)-1)]
            
            coverage = calculate_coverage(
                panel_pos, 
                weather_system.clouds,
                cc_percent
            )
            eff_reduction = coverage * cloud_factor_dict[pname]
            final_gen = baseline_val * (1 - eff_reduction)
            
            panel_generation[pname] = final_gen
            total_gen += final_gen
        
        # Draw UI
        weather_data = {
            'cc': weather_system.cc_5min[min(frame_idx, len(weather_system.cc_5min)-1)],
            'ws': weather_system.wspd_5min[min(frame_idx, len(weather_system.wspd_5min)-1)],
            'wd': weather_system.wdir_5min[min(frame_idx, len(weather_system.wdir_5min)-1)],
            'pattern': weather_system.current_pattern
        }
        viz_system.draw_ui(frame_img, frame_idx, total_gen, weather_data)
        
        im.set_data(np.array(frame_img))
        return [im]
    
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=INTERVAL_MS, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
