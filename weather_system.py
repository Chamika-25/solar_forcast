import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import math
import random
import time
import sim_config as CFG

# Add SCATTER_PROBABILITY to sim_config if not already there
if not hasattr(CFG, 'SCATTER_PROBABILITY'):
    CFG.SCATTER_PROBABILITY = 0.0002

class TurbidityModel:
    """Model atmospheric turbidity variations."""
    
    def __init__(self, location: Dict):
        self.location = location
        self.base_turbidity = self._get_base_turbidity()
        
    def _get_base_turbidity(self) -> float:
        """Get base turbidity for location."""
        # Simplified - could use actual data
        if self.location['altitude'] > 1000:
            return 2.0  # Mountain
        elif abs(self.location['latitude']) > 40:
            return 2.5  # Temperate
        else:
            return 3.5  # Tropical
            
    def get_turbidity(self, timestamp: datetime, weather_state) -> float:
        """Calculate current turbidity."""
        base = self.base_turbidity
        
        # Seasonal variation
        month = timestamp.month
        seasonal_factor = 1 + 0.3 * np.sin((month - 6) * np.pi / 6)
        
        # Humidity effect
        humidity_factor = 1 + 0.01 * (weather_state.humidity - 50)
        
        # After rain effect
        if weather_state.precipitation > 0:
            rain_factor = 0.7  # Cleaner air
        else:
            rain_factor = 1.0
            
        turbidity = base * seasonal_factor * humidity_factor * rain_factor
        
        return max(1.5, min(6.0, turbidity))

class CloudModel:
    """Advanced cloud modeling for weather system."""
    
    def __init__(self):
        self.cloud_database = self._load_cloud_properties()
        
    def _load_cloud_properties(self) -> Dict:
        """Load detailed cloud properties."""
        return {
            'cumulus': {
                'base_height': 500,  # meters
                'thickness': 1000,
                'albedo': 0.65,
                'emissivity': 0.95,
                'liquid_water_path': 50  # g/m²
            },
            'stratus': {
                'base_height': 200,
                'thickness': 500,
                'albedo': 0.60,
                'emissivity': 0.98,
                'liquid_water_path': 100
            },
            'cirrus': {
                'base_height': 8000,
                'thickness': 2000,
                'albedo': 0.30,
                'emissivity': 0.50,
                'liquid_water_path': 5
            },
            'cumulonimbus': {
                'base_height': 500,
                'thickness': 10000,
                'albedo': 0.90,
                'emissivity': 1.0,
                'liquid_water_path': 500
            }
        }
    
    def get_cloud_properties(self, cloud_type: str) -> Dict:
        """Get detailed properties for cloud type."""
        return self.cloud_database.get(cloud_type, self.cloud_database['cumulus'])

@dataclass
class WeatherState:
    """Current weather conditions."""
    timestamp: datetime
    temperature: float  # Celsius
    pressure: float  # hPa
    humidity: float  # %
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    cloud_cover: float  # fraction (0-1)
    cloud_type: str
    visibility: float  # km
    precipitation: float  # mm/hr
    solar_radiation: Dict  # DNI, DHI, GHI measurements

class WeatherSystem:
    """
    Comprehensive weather system that provides realistic atmospheric conditions
    for solar farm simulation.
    """
    
    def __init__(self, location: Dict = None, seed=0):
        """
        Initialize weather system for specific location.
        
        Args:
            location: Dict with 'latitude', 'longitude', 'altitude', 'timezone'
        """
        # Default location (Sri Lanka)
        if location is None:
            location = {
                'latitude': 6.9271,
                'longitude': 79.8612,
                'altitude': 10.0,
                'timezone': 'Asia/Colombo'
            }
            
        self.location = location
        self.current_state = None
        self.forecast = []
        
        # Weather patterns
        self.diurnal_patterns = self._initialize_diurnal_patterns()
        self.seasonal_patterns = self._initialize_seasonal_patterns()
        
        # Atmospheric parameters
        self.turbidity_model = TurbidityModel(location)
        self.cloud_model = CloudModel()
        
        # Import EnhancedWindField here to avoid circular imports
        from enhanced_wind_field import EnhancedWindField
        
        # Initialize wind field for cloud movement
        self.wind = EnhancedWindField()
        
        # Initialize cloud parcels list
        self.parcels = []
        self.sim_time = 0.0
        self.time_since_last_spawn = 0.0
        self.max_gap_sec = 3.0  # max allowed cloud-free interval
        
        # Initialize global weather pattern state
        self.pattern_intensity = 0.5
        self.pattern_direction = 0.0
        self.current_formation_probability = getattr(CFG, 'SPAWN_PROBABILITY', 0.04)
        self.humidity = 60.0
        self.temperature = 28.0
        
    def _initialize_diurnal_patterns(self) -> Dict:
        """Initialize typical daily weather patterns."""
        return {
            'temperature': {
                'min_hour': 6,
                'max_hour': 15,
                'amplitude': 8  # Daily temperature range
            },
            'humidity': {
                'min_hour': 15,
                'max_hour': 6,
                'amplitude': 20
            },
            'wind_speed': {
                'min_hour': 6,
                'max_hour': 15,
                'amplitude': 3
            },
            'cloud_formation': {
                'morning_fog': {'start': 5, 'end': 8, 'probability': 0.3},
                'afternoon_cumulus': {'start': 11, 'end': 17, 'probability': 0.4},
                'evening_clear': {'start': 18, 'end': 22, 'probability': 0.7}
            }
        }
    
    def _initialize_seasonal_patterns(self) -> Dict:
        """Initialize seasonal weather variations."""
        return {
            'summer': {
                'temp_range': (25, 35),
                'humidity_range': (50, 80),
                'cloud_types': ['cumulus', 'cumulonimbus'],
                'turbidity': 4.0
            },
            'monsoon': {
                'temp_range': (22, 30),
                'humidity_range': (70, 95),
                'cloud_types': ['nimbostratus', 'cumulonimbus'],
                'turbidity': 5.0
            },
            'winter': {
                'temp_range': (18, 28),
                'humidity_range': (40, 70),
                'cloud_types': ['cirrus', 'altocumulus'],
                'turbidity': 2.5
            }
        }
    
    def update_weather_pattern(self):
        """Update global weather pattern that influences all clouds"""
        # Weather pattern cycles over time 
        pattern_cycle = (self.sim_time / 3600) % 24  # 24-hour cycle
        
        # Calculate pattern intensity (0-1)
        self.pattern_intensity = 0.5 + 0.5 * math.sin(pattern_cycle * math.pi / 12)
        
        # Update wind directions based on pattern
        pattern_angle = (self.sim_time / 7200) % 360  # Slowly rotating wind pattern
        self.pattern_direction = pattern_angle
        
        # Update temperature and humidity based on time of day
        hour = (self.sim_time / 3600) % 24
        temp_pattern = self.diurnal_patterns['temperature']
        humid_pattern = self.diurnal_patterns['humidity']
        
        # Temperature cycle (cooler at night, warmer in day)
        temp_phase = ((hour - temp_pattern['min_hour']) / 
                      (temp_pattern['max_hour'] - temp_pattern['min_hour']) % 1) * 2 * math.pi
        self.temperature = 28 + 5 * math.sin(temp_phase - math.pi/2)
        
        # Humidity cycle (inverse of temperature)
        humid_phase = ((hour - humid_pattern['min_hour']) / 
                       (humid_pattern['max_hour'] - humid_pattern['min_hour']) % 1) * 2 * math.pi
        self.humidity = 60 + 20 * math.sin(humid_phase - math.pi/2)
        
        # Cloud formation probability varies with pattern
        base_prob = getattr(CFG, 'SPAWN_PROBABILITY', 0.04)
        time_factor = 1.0 + 0.5 * math.sin(pattern_cycle * math.pi / 12)
        
        # More clouds in high humidity
        humidity_factor = 1.0 + (self.humidity - 60) / 100
        
        self.current_formation_probability = base_prob * time_factor * humidity_factor
    
    def update(self, timestamp: datetime) -> WeatherState:
        """Update weather state for given timestamp."""
        # Determine season
        season = self._get_season(timestamp)
        seasonal_params = self.seasonal_patterns[season]
        
        # Calculate base values
        hour = timestamp.hour + timestamp.minute / 60
        
        # Temperature
        temp_min, temp_max = seasonal_params['temp_range']
        temp = self._calculate_diurnal_value(
            hour, temp_min, temp_max,
            self.diurnal_patterns['temperature']
        )
        
        # Humidity (inverse of temperature)
        humidity_min, humidity_max = seasonal_params['humidity_range']
        humidity = self._calculate_diurnal_value(
            hour, humidity_max, humidity_min,  # Note: reversed
            self.diurnal_patterns['humidity']
        )
        
        # Wind
        wind_base = 2.0
        wind_amplitude = self.diurnal_patterns['wind_speed']['amplitude']
        wind_speed = self._calculate_diurnal_value(
            hour, wind_base, wind_base + wind_amplitude,
            self.diurnal_patterns['wind_speed']
        )
        
        # Wind direction (prevailing + random variation)
        wind_direction = 225 + np.random.normal(0, 20)  # SW prevailing
        
        # Pressure (small variations)
        pressure = 1013 + np.random.normal(0, 5)
        
        # Cloud cover and type
        cloud_data = self._determine_clouds(hour, season)
        
        # Visibility
        visibility = self._calculate_visibility(humidity, cloud_data['cloud_cover'])
        
        # Precipitation
        precipitation = self._calculate_precipitation(cloud_data['cloud_type'], humidity)
        
        # Solar radiation (if daytime)
        solar_radiation = self._calculate_solar_radiation(
            timestamp, cloud_data['cloud_cover'], seasonal_params['turbidity']
        )
        
        # Create weather state
        self.current_state = WeatherState(
            timestamp=timestamp,
            temperature=temp,
            pressure=pressure,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_direction=wind_direction % 360,
            cloud_cover=cloud_data['cloud_cover'],
            cloud_type=cloud_data['cloud_type'],
            visibility=visibility,
            precipitation=precipitation,
            solar_radiation=solar_radiation
        )
        
        return self.current_state
    
    def _calculate_diurnal_value(self, hour: float, min_val: float, max_val: float,
                               pattern: Dict) -> float:
        """Calculate value following diurnal pattern."""
        min_hour = pattern['min_hour']
        max_hour = pattern['max_hour']
        
        # Simple sinusoidal model
        if min_hour < max_hour:
            # Normal case (min in morning, max in afternoon)
            phase = (hour - min_hour) / (max_hour - min_hour) * np.pi
        else:
            # Inverted case (min in afternoon, max in morning)
            if hour >= max_hour:
                phase = (hour - max_hour) / (24 - max_hour + min_hour) * np.pi
            else:
                phase = (hour + 24 - max_hour) / (24 - max_hour + min_hour) * np.pi
                
        value = min_val + (max_val - min_val) * (0.5 + 0.5 * np.sin(phase - np.pi/2))
        
        # Add random variation
        value += np.random.normal(0, (max_val - min_val) * 0.05)
        
        return value
    
    def _get_season(self, timestamp: datetime) -> str:
        """Determine season based on date (for tropical location)."""
        month = timestamp.month
        
        if month in [6, 7, 8, 9]:
            return 'monsoon'
        elif month in [3, 4, 5]:
            return 'summer'
        else:
            return 'winter'
    
    def _determine_clouds(self, hour: float, season: str) -> Dict:
        """Determine cloud cover and type based on time and season."""
        cloud_patterns = self.diurnal_patterns['cloud_formation']
        seasonal_clouds = self.seasonal_patterns[season]['cloud_types']
        
        # Check each cloud formation period
        cloud_cover = 0.0
        cloud_type = 'clear'
        
        for period_name, period in cloud_patterns.items():
            if period['start'] <= hour <= period['end']:
                if np.random.random() < period['probability']:
                    if 'fog' in period_name:
                        cloud_cover = np.random.uniform(0.6, 0.9)
                        cloud_type = 'stratus'
                    elif 'cumulus' in period_name:
                        cloud_cover = np.random.uniform(0.3, 0.7)
                        cloud_type = np.random.choice(seasonal_clouds)
                    else:
                        cloud_cover = np.random.uniform(0, 0.3)
                        cloud_type = 'clear'
                        
        return {'cloud_cover': cloud_cover, 'cloud_type': cloud_type}
    
    def _calculate_visibility(self, humidity: float, cloud_cover: float) -> float:
        """Calculate visibility based on humidity and clouds."""
        # Base visibility
        base_vis = 50  # km
        
        # Humidity effect (exponential decay)
        humidity_factor = np.exp(-0.02 * (humidity - 40))
        
        # Cloud effect
        cloud_factor = 1 - 0.5 * cloud_cover
        
        visibility = base_vis * humidity_factor * cloud_factor
        
        return max(1, visibility)
    
    def _calculate_precipitation(self, cloud_type: str, humidity: float) -> float:
        """Calculate precipitation rate."""
        precip_rates = {
            'clear': 0,
            'cirrus': 0,
            'altocumulus': 0,
            'cumulus': 0.1 if humidity > 80 else 0,
            'stratocumulus': 0.5 if humidity > 85 else 0,
            'nimbostratus': 2.0,
            'cumulonimbus': 10.0 if humidity > 75 else 0
        }
        
        base_rate = precip_rates.get(cloud_type, 0)
        
        # Random variation
        if base_rate > 0:
            base_rate *= np.random.uniform(0.5, 1.5)
            
        return base_rate
    
    def _calculate_solar_radiation(self, timestamp: datetime, cloud_cover: float,
                                 turbidity: float) -> Dict:
        """Calculate solar radiation components."""
        # Simple calculation if clear_sky_generation module not available
        hour = timestamp.hour + timestamp.minute / 60
        if hour < 6 or hour > 18:
            # Night time - no radiation
            return {
                'dni': 0, 'dhi': 0, 'ghi': 0, 
                'clear_sky_dni': 0, 'clear_sky_ghi': 0,
                'elevation': 0, 'azimuth': 0
            }
        
        # Approximate solar elevation (0 at sunrise/sunset, max at noon)
        solar_elevation = 70 * np.sin(np.pi * (hour - 6) / 12)
        
        # Simple clear sky model
        max_dni = 1000  # W/m²
        max_dhi = 100   # W/m²
        
        # Clear sky values
        clear_sky_dni = max_dni * (solar_elevation / 90)
        clear_sky_dhi = max_dhi * (solar_elevation / 90)
        clear_sky_ghi = clear_sky_dni * np.sin(np.radians(solar_elevation)) + clear_sky_dhi
        
        # Apply cloud effects
        cloud_factor = 1 - cloud_cover * 0.75
        
        return {
            'dni': clear_sky_dni * cloud_factor,
            'dhi': clear_sky_dhi * (1 + cloud_cover * 0.2),  # Diffuse increases with clouds
            'ghi': clear_sky_ghi * (1 - cloud_cover * 0.5),
            'clear_sky_dni': clear_sky_dni,
            'clear_sky_ghi': clear_sky_ghi,
            'elevation': solar_elevation,
            'azimuth': 180 * (hour - 6) / 12  # 0° at sunrise, 180° at sunset
        }
    
    def _spawn(self, t):
        """
        Spawn a new cloud parcel at the upwind edge of the domain.
        
        Args:
            t: Current simulation time
        """
        # Import CloudParcel here to avoid circular imports
        from cloud_simulation import EnhancedCloudParcel
        
        # Get wind direction from wind field to determine spawn location
        try:
            _, hdg = self.wind.sample(CFG.DOMAIN_SIZE_M/2, CFG.DOMAIN_SIZE_M/2, 1000)
        except:
            hdg = 90  # Default East direction
            
        d = CFG.DOMAIN_SIZE_M
        b = CFG.SPAWN_BUFFER_M
        
        # Spawn at the upwind edge of the domain based on wind direction
        upwind_angle = (hdg + 180) % 360  # Opposite of wind direction
        upwind_rad = math.radians(upwind_angle)
        
        # Calculate spawn position at upwind edge
        edge_distance = 0.1 * d  # 10% inside domain edge
        
        # Calculate distance from center to edge in upwind direction
        center_x = d / 2
        center_y = d / 2
        
        # Calculate which edge to spawn on based on wind direction
        if 45 <= upwind_angle < 135:  # Wind from North, spawn at North edge
            spawn_x = center_x + random.uniform(-0.4, 0.4) * d
            spawn_y = -edge_distance
        elif 135 <= upwind_angle < 225:  # Wind from East, spawn at East edge
            spawn_x = d + edge_distance
            spawn_y = center_y + random.uniform(-0.4, 0.4) * d
        elif 225 <= upwind_angle < 315:  # Wind from South, spawn at South edge
            spawn_x = center_x + random.uniform(-0.4, 0.4) * d
            spawn_y = d + edge_distance
        else:  # Wind from West, spawn at West edge
            spawn_x = -edge_distance
            spawn_y = center_y + random.uniform(-0.4, 0.4) * d
        
        # Select cloud type based on weather conditions
        # Higher humidity favors cumulus and cumulonimbus
        if self.humidity > 80:
            cloud_type_weights = {"cirrus": 0.1, "cumulus": 0.4, "cumulonimbus": 0.5}
        elif self.humidity > 60:
            cloud_type_weights = {"cirrus": 0.3, "cumulus": 0.6, "cumulonimbus": 0.1}
        else:
            cloud_type_weights = {"cirrus": 0.7, "cumulus": 0.2, "cumulonimbus": 0.1}
            
        # Select cloud type based on calculated weights
        cloud_types = list(cloud_type_weights.keys())
        weights = list(cloud_type_weights.values())
        ctype = random.choices(cloud_types, weights=weights)[0]
        
        # Create new cloud parcel with enhanced visualization
        try:
            new_cloud = EnhancedCloudParcel(spawn_x, spawn_y, self.wind, ctype)
            self.parcels.append(new_cloud)
            print(f"Spawned new {ctype} cloud at ({spawn_x:.1f}, {spawn_y:.1f})")
        except ImportError:
            # Fallback if EnhancedCloudParcel not available
            print("Warning: EnhancedCloudParcel not available, cloud spawning disabled")
    
    def step(self, t=None, dt=None, t_s=None):
        """
        Update the weather system for one time step.
        
        Args:
            t: Frame count or time value
            dt: Time step in seconds
            t_s: Simulation time in seconds (optional)
        """
        # Update simulation time
        if t_s is not None:
            self.sim_time = t_s
        elif t is not None:
            self.sim_time = t
        else:
            self.sim_time += 1
        
        # Use physics timestep if dt not provided
        if dt is None:
            dt = CFG.PHYSICS_TIMESTEP if hasattr(CFG, 'PHYSICS_TIMESTEP') else 1/60.0
        
        # Update global weather pattern
        self.update_weather_pattern()
        
        # Update wind field
        if hasattr(self.wind, 'step'):
            self.wind.step(self.sim_time)
        
        # Update all parcels with wind
        expired_indices = []
        for i, parcel in enumerate(self.parcels):
            if hasattr(parcel, 'step'):
                if parcel.step(dt, self.wind, self.sim_time):
                    expired_indices.append(i)
        
        # Remove expired parcels (in reverse to preserve indices)
        for i in sorted(expired_indices, reverse=True):
            if i < len(self.parcels):
                print(f"Removing cloud that has crossed the domain")
                self.parcels.pop(i)
        
        # Handle cloud scattering
        new_parcels = []
        for p in self.parcels:
            if getattr(p, "flag_for_split", False):
                # Reduce number of fragments to reduce disappearance effect
                n = random.randint(2, 3)  # Instead of using CFG.SCATTER_FRAGMENTS
                print(f"Cloud at ({p.x:.1f}, {p.y:.1f}) scattering into {n} fragments")
                
                # Make child clouds more visible
                for _ in range(n):
                    from cloud_simulation import EnhancedCloudParcel
                    child = p.__class__(
                        p.x + random.uniform(-0.5, 0.5) * 1000,
                        p.y + random.uniform(-0.5, 0.5) * 1000,
                        p.wind,
                        p.type)
                    child.vx, child.vy = p.vx, p.vy
                    # Make child clouds larger (0.8-0.9 of parent instead of 0.5-0.8)
                    child.r = p.r * random.uniform(0.8, 0.9)
                    # Start children in stable phase
                    child.age = p.growth_frames
                    new_parcels.append(child)
                
                p.flag_for_split = False  # Reset flag
                p.split_fading = 60   # keep parent 60 frames (≈1 s)
                new_parcels.append(p) # parent will fade in its own step()
            else:
                new_parcels.append(p)
        self.parcels = new_parcels
        
        # Spawn new clouds if needed
        # In single-cloud mode, only spawn if there are no clouds
        if hasattr(CFG, 'SINGLE_CLOUD_MODE') and CFG.SINGLE_CLOUD_MODE:
            if len(self.parcels) == 0:
                self._spawn(self.sim_time)
                self.time_since_last_spawn = 0.0
        else:
            # Update time since last spawn
            self.time_since_last_spawn += dt
            
            # Check if we need a forced spawn due to empty sky for too long
            need_forced = (len(self.parcels) == 0 and
                          self.time_since_last_spawn > self.max_gap_sec)
            
            # Regular spawning behavior based on MAX_PARCELS
            max_parcels = CFG.MAX_PARCELS if hasattr(CFG, 'MAX_PARCELS') else 12
            spawn_prob = self.current_formation_probability
            
            if len(self.parcels) < max_parcels and (
                    random.random() < spawn_prob or need_forced):
                self._spawn(self.sim_time)
                self.time_since_last_spawn = 0.0
                
            # Handle forced initial cloud
            if len(self.parcels) == 0 and hasattr(CFG, 'FORCE_INITIAL_CLOUD') and CFG.FORCE_INITIAL_CLOUD:
                self._spawn(self.sim_time)
                self.time_since_last_spawn = 0.0
    
    def get_avg_trajectory(self):
        """Get average trajectory (speed and direction) of all cloud parcels."""
        if not self.parcels:
            return None, None, 0
            
        speeds = []
        directions_x = []
        directions_y = []
        
        for parcel in self.parcels:
            if hasattr(parcel, 'vx') and hasattr(parcel, 'vy'):
                speed = math.sqrt(parcel.vx**2 + parcel.vy**2)
                direction = math.degrees(math.atan2(parcel.vy, parcel.vx)) % 360
                speeds.append(speed)
                direction_rad = math.radians(direction)
                directions_x.append(math.cos(direction_rad))
                directions_y.append(math.sin(direction_rad))
        
        if not speeds:
            return None, None, 0
            
        avg_speed = sum(speeds) / len(speeds)
        avg_dir_x = sum(directions_x) / len(directions_x)
        avg_dir_y = sum(directions_y) / len(directions_y)
        avg_direction = math.degrees(math.atan2(avg_dir_y, avg_dir_x)) % 360
        mean_vector_length = math.sqrt(avg_dir_x**2 + avg_dir_y**2)
        confidence = mean_vector_length
        
        # Convert m/s to km/h
        avg_speed_kmh = avg_speed * 3.6
        return avg_speed_kmh, avg_direction, confidence
    
    def current_cloud_cover_pct(self):
        """Calculate percentage of domain covered by clouds."""
        if not self.parcels:
            return 0.0
            
        total_area = 0
        domain_area = CFG.AREA_SIZE_KM * CFG.AREA_SIZE_KM if hasattr(CFG, 'AREA_SIZE_KM') else 50*50
        
        for p in self.parcels:
            if hasattr(p, 'r'):
                cloud_area = math.pi * (p.r ** 2)
                total_area += cloud_area
                
        cover = min(100, (total_area / domain_area) * 100 * 5)
        return cover
    
    def get_forecast(self, hours_ahead: int = 24) -> List[WeatherState]:
        """Generate weather forecast."""
        forecast = []
        current = datetime.now()
        
        for hour in range(hours_ahead):
            future_time = current + timedelta(hours=hour)
            weather = self.update(future_time)
            forecast.append(weather)
            
        return forecast
    
    def get_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate historical weather data."""
        data = []
        current = start_date
        
        while current <= end_date:
            weather = self.update(current)
            data.append({
                'timestamp': weather.timestamp,
                'temperature': weather.temperature,
                'humidity': weather.humidity,
                'wind_speed': weather.wind_speed,
                'wind_direction': weather.wind_direction,
                'cloud_cover': weather.cloud_cover,
                'dni': weather.solar_radiation['dni'],
                'dhi': weather.solar_radiation['dhi'],
                'ghi': weather.solar_radiation['ghi']
            })
            current += timedelta(minutes=5)
            
        return pd.DataFrame(data)