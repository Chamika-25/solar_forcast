import math
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
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
MAX_CLOUDS = 100            # Maximum number of clouds in the simulation
COVERAGE_THRESHOLD = 0.05   # Lower threshold for gradual changes

# Season setting (affects cloud behavior)
SEASON = "SUMMER"  # Options: "SUMMER", "WINTER", "SPRING", "FALL"

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

# Cloud size parameters with more variety
CLOUD_SIZES = {
    "TINY": {
        "width_range": (20, 35),
        "height_range": (15, 25),
        "probability": 0.2,     # 20% chance of tiny clouds
        "scale_factor": 0.4     # Smaller rendering scale
    },
    "SMALL": {
        "width_range": (40, 80),
        "height_range": (25, 50),
        "probability": 0.4,     # 40% chance of small clouds
        "scale_factor": 0.5
    },
    "MEDIUM": {
        "width_range": (90, 130),
        "height_range": (50, 75),
        "probability": 0.3,     # 30% chance of medium clouds
        "scale_factor": 0.6
    },
    "LARGE": {
        "width_range": (140, 200),
        "height_range": (80, 120),
        "probability": 0.1,     # 10% chance of large clouds
        "scale_factor": 0.7
    }
}

# Default values (backward compatibility)
CLOUD_WIDTH_MIN = 40
CLOUD_WIDTH_MAX = 80
CLOUD_HEIGHT_MIN = 25
CLOUD_HEIGHT_MAX = 100
CLOUD_SCALE_FACTOR = 0.5

##############################
# 2. ENHANCED CLOUD CLASS
##############################
class Cloud:
    def __init__(self, birth_frame, size_factor=1.0, position=None, cloud_size=None):
        # Determine cloud size category 
        if cloud_size is None:
            # Select random size based on probabilities
            r = random.random()
            cum_prob = 0
            for size, params in CLOUD_SIZES.items():
                cum_prob += params["probability"]
                if r <= cum_prob:
                    cloud_size = size
                    break
            # Default to SMALL if something goes wrong
            if cloud_size is None:
                cloud_size = "SMALL"
        
        # Get size parameters for this cloud type
        size_params = CLOUD_SIZES[cloud_size]
        self.cloud_size = cloud_size
        
        # Apply size factor to the base ranges
        width_min = int(size_params["width_range"][0] * size_factor)
        width_max = int(size_params["width_range"][1] * size_factor)
        height_min = int(size_params["height_range"][0] * size_factor)
        height_max = int(size_params["height_range"][1] * size_factor)
        
        # Set scale factor based on cloud size
        self.scale_factor = size_params["scale_factor"]
        
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
        
        # Store original dimensions for wind effects
        self.original_width = self.width
        self.original_height = self.height
        
    def update(self, dx, dy, frame_idx, wind_speed=0):
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
        
        # Wind stretching effects
        if wind_speed > 3.0:
            # Stretch cloud based on wind speed (stronger wind = more stretching)
            stretch = 1.0 + (wind_speed / 10.0) * 0.3
            self.width = min(int(self.width * stretch), int(CLOUD_WIDTH_MAX * 1.5))
            
            # Keep aspect ratio reasonable
            if self.width > self.height * 2:
                self.height = int(self.width / 2)
        else:
            # Gradually return to original shape when wind is low
            if abs(self.width - self.original_width) > 2:
                self.width = int(self.width * 0.95 + self.original_width * 0.05)
            if abs(self.height - self.original_height) > 2:
                self.height = int(self.height * 0.95 + self.original_height * 0.05)

##############################
# 3. ENHANCED WEATHER SYSTEM
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
        
        # Apply seasonal adjustments
        self.apply_seasonal_adjustments()
        
        self.clouds = []
        
        # Cloud pattern control
        self.current_pattern = "SCATTERED"  # Start with scattered pattern
        self.pattern_change_frame = 0
        self.next_pattern_change = random.randint(*CLOUD_PATTERNS["SCATTERED"]["duration"])
        self.target_cloud_count = random.randint(*CLOUD_PATTERNS["SCATTERED"]["count_range"])
        
        # Weather state variables
        self.atmospheric_stability = 0.5  # 0-1 scale (lower = less stable = more active)
        self.is_precipitating = False
        
    def apply_seasonal_adjustments(self):
        """Apply seasonal effects to base weather parameters"""
        if SEASON == "WINTER":
            # Winter has stronger winds, less cloud cover
            self.wspd_5min = np.clip(self.wspd_5min * 1.2, 0, 15)
            self.cc_5min = np.clip(self.cc_5min * 0.7, 0, 100)
        elif SEASON == "SUMMER":
            # Summer has more cloud cover variation
            variation = np.sin(np.linspace(0, 10*np.pi, len(self.cc_5min))) * 15
            self.cc_5min = np.clip(self.cc_5min + variation, 0, 100)
        elif SEASON == "SPRING":
            # Spring has more variability in wind and cloud cover
            variation = np.sin(np.linspace(0, 15*np.pi, len(self.cc_5min))) * 20
            self.cc_5min = np.clip(self.cc_5min + variation, 0, 100)
            
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
    
    def update_atmospheric_conditions(self, frame_idx):
        """Update atmospheric stability based on time of day"""
        hour = (frame_idx // FRAMES_PER_HOUR) % 24
        
        # Stability varies by time of day (lower = less stable = more cloud activity)
        if 9 <= hour < 15:  # Day time - less stable due to heating
            target_stability = 0.3
        elif 19 <= hour or hour < 5:  # Night - more stable
            target_stability = 0.7
        else:  # Morning/evening transitions
            target_stability = 0.5
            
        # Gradually adjust stability (slow transitions)
        if self.atmospheric_stability < target_stability:
            self.atmospheric_stability = min(self.atmospheric_stability + 0.01, target_stability)
        else:
            self.atmospheric_stability = max(self.atmospheric_stability - 0.01, target_stability)
    
    def select_new_pattern(self):
        """Choose next cloud pattern based on natural transitions"""
        # Patterns tend to evolve naturally rather than randomly jump
        if self.current_pattern == "SCATTERED":
            # From scattered, most likely to become bunched or stay scattered
            pattern_probs = {"SCATTERED": 0.6, "BUNCHED": 0.3, "ISOLATED": 0.1}
        elif self.current_pattern == "BUNCHED":
            # From bunched, most likely to become scattered or stay bunched
            pattern_probs = {"SCATTERED": 0.4, "BUNCHED": 0.5, "ISOLATED": 0.1}
        else:  # ISOLATED
            # From isolated, most likely to stay isolated or become scattered
            pattern_probs = {"SCATTERED": 0.3, "BUNCHED": 0.1, "ISOLATED": 0.6}
        
        # Select pattern based on probabilities
        r = random.random()
        cum_prob = 0
        for pattern, prob in pattern_probs.items():
            cum_prob += prob
            if r <= cum_prob:
                self.current_pattern = pattern
                break
        
        # Set parameters for this pattern
        pattern_params = CLOUD_PATTERNS[self.current_pattern]
        self.target_cloud_count = random.randint(*pattern_params["count_range"])
        
        # Pattern duration affected by atmospheric stability
        if self.atmospheric_stability < 0.4:  # Unstable - faster changes
            self.next_pattern_change = random.randint(
                max(pattern_params["duration"][0] - 10, 15),
                max(pattern_params["duration"][1] - 15, 30)
            )
        else:  # More stable - slower changes
            self.next_pattern_change = random.randint(
                pattern_params["duration"][0],
                pattern_params["duration"][1]
            )
    
    def apply_time_of_day_effects(self, frame_idx):
        """Apply time-of-day effects to clouds"""
        hour = (frame_idx // FRAMES_PER_HOUR) % 24
        
        for cloud in self.clouds:
            if not cloud.active:
                continue
                
            # Morning (rising thermals can break up clouds)
            if 7 <= hour < 11 and random.random() < 0.01:
                cloud.opacity *= 0.8
                
            # Afternoon (clouds can grow with heat)
            elif 13 <= hour < 17 and random.random() < 0.02:
                cloud.width = min(int(cloud.width * 1.1), CLOUD_WIDTH_MAX * 1.5)
                cloud.height = min(int(cloud.height * 1.1), CLOUD_HEIGHT_MAX * 1.5)
    
    def handle_cloud_merging(self):
        """Handle clouds merging when they get close to each other"""
        if len(self.clouds) <= 1:
            return
            
        merged_indices = []
        
        for i, cloud1 in enumerate(self.clouds):
            if i in merged_indices:
                continue
                
            for j, cloud2 in enumerate(self.clouds[i+1:], i+1):
                if j in merged_indices or not cloud1.active or not cloud2.active:
                    continue
                    
                # Calculate centers and distance
                c1_x, c1_y = cloud1.x + cloud1.width/2, cloud1.y + cloud1.height/2
                c2_x, c2_y = cloud2.x + cloud2.width/2, cloud2.y + cloud2.height/2
                distance = math.hypot(c1_x - c2_x, c1_y - c2_y)
                
                # Merge threshold based on cloud sizes
                threshold = (cloud1.width + cloud2.width) / 3
                
                # Merge clouds if close enough
                if distance < threshold and random.random() < 0.3:  # 30% chance when close
                    # Grow first cloud
                    cloud1.width = min(int(cloud1.width * 1.3), CLOUD_WIDTH_MAX * 2)
                    cloud1.height = min(int(cloud1.height * 1.2), CLOUD_HEIGHT_MAX * 2)
                    
                    # Update original dimensions
                    cloud1.original_width = cloud1.width
                    cloud1.original_height = cloud1.height
                    
                    # Mark second cloud for removal
                    merged_indices.append(j)
        
        # Remove merged clouds
        self.clouds = [c for i, c in enumerate(self.clouds) if i not in merged_indices]
    
    def create_clouds(self, frame_idx):
        # Check if it's time to change the pattern
        if frame_idx - self.pattern_change_frame >= self.next_pattern_change:
            self.pattern_change_frame = frame_idx
            self.select_new_pattern()
            print(f"Frame {frame_idx}: Changing to {self.current_pattern} pattern, target {self.target_cloud_count} clouds")
        
        # Add clouds to match the target count
        current_count = len(self.clouds)
        
        if current_count < self.target_cloud_count and current_count < MAX_CLOUDS:
            pattern_params = CLOUD_PATTERNS[self.current_pattern]
            grouping_factor = pattern_params["grouping_factor"]
            
            # Either add to existing cluster or create new cloud
            if random.random() < grouping_factor and current_count > 0:
                # Add cloud near an existing cloud
                parent_cloud = random.choice(self.clouds)
                position = (parent_cloud.x, parent_cloud.y)
                size_factor = 0.8 + random.random() * 0.4
                
                self.clouds.append(Cloud(frame_idx, size_factor, position))
            else:
                # Create a completely new cloud
                size_factor = 0.7 + random.random() * 0.6
                self.clouds.append(Cloud(frame_idx, size_factor))
    
    def update_clouds(self, frame_idx):
        # Update atmospheric conditions based on time of day
        self.update_atmospheric_conditions(frame_idx)
        
        # Create or update cloud patterns
        self.create_clouds(frame_idx)
        
        # Apply time-of-day effects
        self.apply_time_of_day_effects(frame_idx)
        
        # Handle cloud merging
        self.handle_cloud_merging()
            
        # Remove expired clouds
        self.clouds = [c for c in self.clouds if c.lifetime < 3600]
        
        # If we have too many clouds, remove oldest ones
        while len(self.clouds) > self.target_cloud_count:
            oldest_idx = 0
            oldest_lifetime = -1
            
            for i, cloud in enumerate(self.clouds):
                if cloud.lifetime > oldest_lifetime:
                    oldest_lifetime = cloud.lifetime
                    oldest_idx = i
            
            if oldest_idx < len(self.clouds):
                self.clouds.pop(oldest_idx)
        
        # Get current weather parameters
        ws = self.wspd_5min[min(frame_idx, len(self.wspd_5min)-1)]
        wd = self.wdir_5min[min(frame_idx, len(self.wdir_5min)-1)]
        
        # Calculate cloud movement vectors
        wd_rad = math.radians(wd)
        dist_km = ws * 0.06 * 5 * CLOUD_SPEED_SCALE
        px_per_km = IMAGE_PIXELS / AREA_SIZE_KM
        dx = dist_km * math.cos(wd_rad) * px_per_km
        dy = -dist_km * math.sin(wd_rad) * px_per_km
        
        # Update all clouds with wind speed
        for cloud in self.clouds:
            cloud.update(dx, dy, frame_idx, ws)
    
    def get_current_weather(self, frame_idx):
        """Return current weather parameters for the given frame"""
        cc = self.cc_5min[min(frame_idx, len(self.cc_5min)-1)]
        ws = self.wspd_5min[min(frame_idx, len(self.wspd_5min)-1)]
        wd = self.wdir_5min[min(frame_idx, len(self.wdir_5min)-1)]
        
        return {
            'cc': cc,
            'ws': ws,
            'wd': wd,
            'pattern': self.current_pattern,
            'stability': self.atmospheric_stability
        }

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
    def __init__(self, panels):
        self.panels = panels
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

        for p in self.panels:
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
            
            # Define cloud parameters using cloud's scale factor
            cloud_center_x = cloud.x + cloud.width/2
            cloud_center_y = cloud.y + cloud.height/2
            base_radius = min(cloud.width, cloud.height) * 0.25 * cloud.scale_factor
            
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

# Helper function to get time string from frame index
def get_time_string(frame_idx):
    minutes = 6*60 + 20 + frame_idx*5
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"
