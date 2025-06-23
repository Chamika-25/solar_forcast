import math
import random
import numpy as np
import time
import sim_config as CFG
from enhanced_wind_field import EnhancedWindField

# --- Cloud Type Appearance Profiles ------------------------
CLOUD_TYPE_PROFILES = {
    "cirrus": {
        "count": 1,
        "cw_range": (3000, 5000),
        "ch_range": (400, 800),
        "opacity_range": (0.2, 0.4),
        "rotation_range": (-0.4, 0.4)
    },
    "cumulus": {
        "count": (3, 5),
        "cw_range": (1200, 2000),
        "ch_range": (1200, 2000),
        "opacity_range": (0.5, 0.7),
        "rotation_range": (-0.1, 0.1)
    },
    "cumulonimbus": {
        "count": (8, 12),
        "cw_range": (1500, 2500),
        "ch_range": (2500, 4500),
        "opacity_range": (0.8, 1.0),
        "rotation_range": (-0.05, 0.05)
    }
}

def _altitude_to_index(alt_km, layer_heights):
    """
    Map altitude in km to a layer index.
    
    Args:
        alt_km: Altitude in kilometers
        layer_heights: List of layer boundary heights in km
        
    Returns:
        Layer index (0 for lowest layer, N-2 for highest layer)
    """
    for i in range(len(layer_heights) - 1):
        if alt_km >= layer_heights[i] and alt_km < layer_heights[i+1]:
            return i
    return len(layer_heights) - 2  # Return the highest layer index

def generate_ellipses_for_type(cloud_type, center_x, center_y):
    profile = CLOUD_TYPE_PROFILES[cloud_type]
    count = profile["count"]
    if isinstance(count, tuple):
        puff_count = random.randint(*count)
    else:
        puff_count = count
    
    ellipses = []
    
    # Handle specific cloud types differently
    if cloud_type == "cirrus":
        # Single long, thin, rotated ellipse
        cw = random.randint(*profile["cw_range"])
        ch = random.randint(*profile["ch_range"])
        crot = random.uniform(*profile["rotation_range"])
        cop = random.uniform(*profile["opacity_range"])
        ellipses.append((center_x, center_y, cw, ch, crot, cop))
    
    elif cloud_type == "cumulus":
        # 3-5 fluffy, round clustered ellipses
        for _ in range(puff_count):
            # Random offsets for clustering
            dx = random.uniform(-500, 500)
            dy = random.uniform(-500, 500)
            cx = center_x + dx
            cy = center_y + dy
            cw = random.randint(*profile["cw_range"])
            ch = random.randint(*profile["ch_range"])
            crot = random.uniform(*profile["rotation_range"])
            cop = random.uniform(*profile["opacity_range"])
            ellipses.append((cx, cy, cw, ch, crot, cop))
    
    elif cloud_type == "cumulonimbus":
        # 8-12 vertically extended ellipses
        y_offsets = np.linspace(-2000, 2000, puff_count)
        for i, y_offset in enumerate(y_offsets):
            # Horizontal position varies less at top (anvil shape)
            x_range = 600 if i < puff_count//2 else 300
            dx = random.uniform(-x_range, x_range)
            cx = center_x + dx
            cy = center_y + y_offset
            
            # Width increases at top for anvil effect
            width_factor = 1.0 if i < puff_count//2 else 1.5
            cw = random.randint(*profile["cw_range"]) * width_factor
            
            # Height decreases at top
            height_factor = 1.0 if i < puff_count//2 else 0.7
            ch = random.randint(*profile["ch_range"]) * height_factor
            
            crot = random.uniform(*profile["rotation_range"])
            
            # Opacity varies - darker at bottom, lighter at top
            opacity_factor = 1.0 if i < puff_count//2 else 0.8
            cop = random.uniform(*profile["opacity_range"]) * opacity_factor
            
            ellipses.append((cx, cy, cw, ch, crot, cop))
    
    return ellipses

def generate_shifted_ellipses(source_puffs, dx, dy, duration):
    shifted = []
    for puff in source_puffs:
        cx, cy, cw, ch, crot, cop = puff
        new_cx = cx + dx * duration * 1000  # Convert km to meters
        new_cy = cy + dy * duration * 1000
        shifted.append((new_cx, new_cy, cw, ch, crot, cop))
    return shifted

def interpolate_ellipses(src, tgt, t):
    # Blend only up to the shorter list's length to avoid IndexError
    n = min(len(src), len(tgt))
    return [
        (
            src[i][0]*(1-t) + tgt[i][0]*t,
            src[i][1]*(1-t) + tgt[i][1]*t,
            src[i][2]*(1-t) + tgt[i][2]*t,
            src[i][3]*(1-t) + tgt[i][3]*t,
            src[i][4]*(1-t) + tgt[i][4]*t,
            src[i][5]*(1-t) + tgt[i][5]*t,
        )
        for i in range(n)
    ]

# metres-per-frame scale  (Δt × domain-km)
M_PER_FRAME = CFG.PHYSICS_TIMESTEP * (CFG.DOMAIN_SIZE_M / 1000.0)

class CloudParcel:
    def __init__(self, x, y, wind, ctype):
        self.x, self.y = x, y
        self.type = ctype
        preset = CFG.CLOUD_TYPES[ctype]
        self.alt = preset["alt_km"]
        r_lo, r_hi = preset["r_km"]
        self.r = random.uniform(r_lo, r_hi)
        self.max_op = preset["opacity_max"]
        self.speed_k = preset["speed_k"]
        self.opacity = 0.0
        self.vx = self.vy = 0.0
        self.wind = wind
        self.position_history_x = []
        self.position_history_y = []
        self.flag_for_split = False  # Flag for cloud scattering
        
        # Cloud lifecycle parameters
        self.age = 0  # Current age in frames
        self.growth_frames = getattr(CFG, 'CLOUD_GROWTH_FRAMES', 3600)  # 60s growth phase
        self.stable_frames = getattr(CFG, 'CLOUD_STABLE_FRAMES', 7200)  # 120s stable phase
        self.decay_frames = getattr(CFG, 'CLOUD_DECAY_FRAMES', 3600)    # 60s decay phase
        self.max_age = self.growth_frames + self.stable_frames + self.decay_frames
        
        # Initialize transformation timers
        self.transformation_cooldown = 0
        
        # For thermal updrafts and vertical movement
        self.vertical_velocity = 0.0
        
        # For debugging
        print(f"DEBUG: Created new {ctype} cloud at ({x}, {y})")

    def _update_velocity(self, timestep_sec, sim_time=None):
        """Update cloud velocity with realistic physics"""
        # Calculate thermal updrafts and buoyancy effects
        # These create vertical movement and affect horizontal flow
        thermal_strength = 0.05 * math.sin(self.x / 5000) * math.sin(self.y / 5000)
        buoyancy = (1.5 - self.alt) * 0.02  # Lower clouds rise, higher clouds sink slightly
        
        # Cloud type affects vertical movement
        if self.type == "cumulus":
            thermal_factor = 1.0
        elif self.type == "cumulonimbus":
            thermal_factor = 1.5  # Stronger vertical movement
        else:
            thermal_factor = 0.2  # Cirrus clouds are less affected
        
        # Apply vertical movement that affects horizontal flow slightly
        vertical_acceleration = thermal_strength * thermal_factor * buoyancy
        self.vertical_velocity += vertical_acceleration * timestep_sec
        
        # Damping to prevent excessive vertical speed
        self.vertical_velocity *= 0.99
        
        # Update altitude based on vertical velocity
        self.alt = max(0.2, min(8.0, self.alt + self.vertical_velocity * timestep_sec))
        
        # Get speed (m/s) and heading (degrees) from wind field at this altitude
        speed, heading = self.wind.sample(self.x, self.y, self.alt * 1000)
        
        # Apply cloud type speed factor
        speed *= self.speed_k
        
        # Convert from meteorological (0° = North) to mathematical convention
        heading_rad = math.radians(heading)
        
        # Convert to velocity components
        vx_new = speed * math.cos(heading_rad) * M_PER_FRAME
        vy_new = speed * math.sin(heading_rad) * M_PER_FRAME
        
        # Apply smoothing
        self.vx = CFG.WIND_SMOOTH * vx_new + (1 - CFG.WIND_SMOOTH) * self.vx
        self.vy = CFG.WIND_SMOOTH * vy_new + (1 - CFG.WIND_SMOOTH) * self.vy

    def update(self, t, hum):
        # For backward compatibility - now calls step with default parameters
        return self.step(CFG.PHYSICS_TIMESTEP, self.wind)

    def step(self, timestep_sec, wind_field=None, sim_time=None):
        """
        Update cloud parcel position based on wind velocity.
        
        Args:
            timestep_sec: Time step in seconds
            wind_field: Wind field object providing velocity
            sim_time: Optional simulation time
            
        Returns:
            Boolean indicating if the cloud should be removed
        """
        # smooth fade-out for a splitting parent
        if getattr(self, "split_fading", 0):
            self.split_fading -= 1
            self.opacity *= 0.9
            if self.split_fading == 0:
                return True            # remove after fade
                
        # Increment age
        self.age += 1
        
        # Decrement transformation cooldown if active
        if self.transformation_cooldown > 0:
            self.transformation_cooldown -= 1
        
        # Store position history
        if len(self.position_history_x) >= CFG.POSITION_HISTORY_LENGTH:
            self.position_history_x.pop(0)
            self.position_history_y.pop(0)
        self.position_history_x.append(self.x)
        self.position_history_y.append(self.y)
        
        # Update velocity with enhanced physics
        self._update_velocity(timestep_sec, sim_time)
        
        # Apply velocity to position
        self.x += self.vx * CFG.MOVEMENT_MULTIPLIER
        self.y += self.vy * CFG.MOVEMENT_MULTIPLIER
        
        # Handle domain wrapping and cloud recycling
        d = CFG.DOMAIN_SIZE_M
        b = CFG.SPAWN_BUFFER_M
        domain_margin = 0.4 * CFG.DOMAIN_SIZE_M  # 40% margin
        
        # Cloud recycling - if the cloud exits the domain, respawn it on the opposite side
        if (self.x > d + domain_margin or self.x < -domain_margin or 
            self.y > d + domain_margin or self.y < -domain_margin):
            
            # Reset the cloud's properties for a "new" cloud
            preset = CFG.CLOUD_TYPES[self.type]
            r_lo, r_hi = preset["r_km"]
            
            # Move to the opposite side with some randomization
            if self.x > d + domain_margin:
                self.x = -domain_margin * random.uniform(0.8, 1.2)
                self.y = random.uniform(0.1, 0.9) * d
            elif self.x < -domain_margin:
                self.x = d + domain_margin * random.uniform(0.8, 1.2)
                self.y = random.uniform(0.1, 0.9) * d
            elif self.y > d + domain_margin:
                self.y = -domain_margin * random.uniform(0.8, 1.2)
                self.x = random.uniform(0.1, 0.9) * d
            elif self.y < -domain_margin:
                self.y = d + domain_margin * random.uniform(0.8, 1.2)
                self.x = random.uniform(0.1, 0.9) * d
                
            # Reset cloud parameters for a new lifecycle
            self.age = 0
            self.r = random.uniform(r_lo, r_hi)
            self.opacity = 0.0
            self.transformation_cooldown = 300  # Prevent immediate transformation
            
            print(f"DEBUG: Recycled cloud at ({self.x/1000:.1f}, {self.y/1000:.1f}) km")
            return False  # Don't remove the cloud
        
        # Apply continuous cloud evolution instead of discrete phases
        altitude_factor = self.alt / 3.0  # Higher clouds evolve more slowly
        evolution_rate = 0.0003 / altitude_factor
        
        # Natural size fluctuation
        size_evolution = math.sin(self.age * 0.01) * 0.2
        humidity_factor = 1.0  # Would ideally come from weather system
        
        # Continuous size changes
        size_change = evolution_rate * size_evolution * humidity_factor
        self.r = self.r * (1 + size_change)
        self.r = max(0.5, min(self.r, CFG.R_MAX_KM if hasattr(CFG, 'R_MAX_KM') else 6.0))
        
        # Dynamic opacity based on cloud maturity and time of day
        day_cycle = (sim_time / 3600) % 24 if sim_time else (self.age / 3600) % 24
        time_factor = 0.8 + 0.2 * math.sin(day_cycle * math.pi / 12)
        
        # Cloud lifecycle opacity
        if self.age < self.growth_frames:
            # Growth phase: increasing opacity
            maturity = self.age / self.growth_frames
        elif self.age < self.growth_frames + self.stable_frames:
            # Stable phase: full opacity
            maturity = 1.0
        else:
            # Decay phase: decreasing opacity
            decay_progress = (self.age - (self.growth_frames + self.stable_frames)) / self.decay_frames
            maturity = 1.0 - min(1.0, decay_progress)
        
        # Calculate final opacity
        self.opacity = min(0.95, max(0.05, self.max_op * maturity * time_factor))
        
        # Handle cloud type transformation with cooldown
        if self.transformation_cooldown <= 0:
            transformation_chance = 0.0005  # Base chance
            
            # Different transformation rules based on cloud type
            if self.type == "cumulus" and self.r > 1.5 and random.random() < transformation_chance:
                # Cumulus clouds can become cumulonimbus when they grow large
                self.type = "cumulonimbus"
                self.max_op = CFG.CLOUD_TYPES["cumulonimbus"]["opacity_max"]
                self.speed_k = CFG.CLOUD_TYPES["cumulonimbus"]["speed_k"]
                self.transformation_cooldown = 1800  # 30 sec cooldown
                print(f"DEBUG: Cloud transformed: cumulus → cumulonimbus")
                
            elif self.type == "cumulonimbus" and self.age > 7200 and random.random() < transformation_chance:
                # Old cumulonimbus clouds can transform to cirrus
                self.type = "cirrus"
                self.alt = CFG.CLOUD_TYPES["cirrus"]["alt_km"]
                self.max_op = CFG.CLOUD_TYPES["cirrus"]["opacity_max"]
                self.speed_k = CFG.CLOUD_TYPES["cirrus"]["speed_k"]
                self.transformation_cooldown = 1800  # 30 sec cooldown
                print(f"DEBUG: Cloud transformed: cumulonimbus → cirrus")
        
        # Periodically log position (every 300 frames)
        if self.age % 300 == 0:
            print(f"DEBUG: Cloud at ({self.x:.1f}, {self.y:.1f}) with opacity {self.opacity:.2f}")
        
        # Cloud splitting logic
        if self.age > self.growth_frames and random.random() < 0.0001:
            # Smaller chance of splitting for more realistic behavior
            self.flag_for_split = True
        
        # Never return True unless for fading parent clouds
        return False

    def ellipse(self):
        diam = self.r * 2000
        return (self.x, self.y, diam, diam, 0, self.opacity, self.alt, self.type)

class EnhancedCloudParcel(CloudParcel):
    def __init__(self, x, y, wind, ctype):
        super().__init__(x, y, wind, ctype)
        self.cloud_type = ctype  # Store cloud type
        self.spawn_x = x  # Track original spawn location
        self.spawn_y = y
        
        # Generate initial puffs
        self.source_puffs = generate_ellipses_for_type(self.cloud_type, self.x, self.y)
        
        # Initialize with same target as source to prevent visual jumps
        self.target_puffs = self.source_puffs[:]
        
        # Initialize interpolation timer
        self.t = 0.0
        self.t_duration = 10.0  # seconds to cross
        
        # Scale lifespan based on cloud size
        size_factor = 1.0
        r_lo, r_hi = CFG.CLOUD_TYPES[ctype]["r_km"]
        avg_size = (r_lo + r_hi) / 2
        # Normalize to make average clouds have factor 1.0
        size_factor = max(1.0, avg_size / 1.0)
        
        # Adjust lifespans based on cloud size
        self.growth_frames = int(self.growth_frames * size_factor)
        self.stable_frames = int(self.stable_frames * size_factor)
        self.decay_frames = int(self.decay_frames * size_factor)
        self.max_age = self.growth_frames + self.stable_frames + self.decay_frames
    
    def step(self, timestep_sec, wind_field=None, sim_time=None):
        """
        Update cloud parcel position based on wind velocity.
        
        Args:
            timestep_sec: Time step in seconds
            wind_field: Optional wind field for custom vectors
            sim_time: Optional simulation time
            
        Returns:
            Boolean indicating if the cloud should be removed
        """
        # smooth fade-out for a splitting parent
        if getattr(self, "split_fading", 0):
            self.split_fading -= 1
            self.opacity *= 0.9
            if self.split_fading == 0:
                return True            # remove after fade
        
        # Call parent implementation for main logic
        result = super().step(timestep_sec, wind_field, sim_time)
        
        # Update cloud shape interpolation
        self.t += timestep_sec / self.t_duration
        if self.t >= 1.0:
            # Reset interpolation
            self.t = 0.0
            
            # Current puffs become source puffs
            self.source_puffs = self.get_current_puffs()
            
            # If the cloud type has changed, update the puffs
            self.target_puffs = generate_ellipses_for_type(self.type, self.x, self.y)
            
            # When recycling a cloud, update spawn location
            if getattr(self, "recycled", False):
                self.spawn_x = self.x
                self.spawn_y = self.y
                self.recycled = False
        
        # If the cloud was recycled, mark for puff update
        if self.x < -0.3 * CFG.DOMAIN_SIZE_M or self.x > 1.3 * CFG.DOMAIN_SIZE_M or \
           self.y < -0.3 * CFG.DOMAIN_SIZE_M or self.y > 1.3 * CFG.DOMAIN_SIZE_M:
            # Mark as recycled to update spawn position on next interpolation
            self.recycled = True
                
        return result
    
    def get_current_puffs(self):
        """Return interpolated puffs adjusted by parcel motion"""
        # Interpolate between source and target puffs (t ∈ [0,1])
        puffs = interpolate_ellipses(self.source_puffs, self.target_puffs, self.t)
        
        # Determine movement offset from original puff center to current parcel position
        # Assumes original center was the first puff's spawn point
        dx = self.x - self.spawn_x
        dy = self.y - self.spawn_y
        
        # Apply offset to every puff
        adjusted = []
        for puff in puffs:
            cx, cy, cw, ch, crot, cop = puff
            adjusted.append((cx + dx, cy + dy, cw, ch, crot, cop))
        
        return adjusted
    
    def get_ellipses(self):
        """Return list of ellipses with altitude and type appended"""
        result = []
        current_puffs = self.get_current_puffs()
        for e in current_puffs:
            cx, cy, cw, ch, crot, cop = e
            # Append altitude and cloud type to each ellipse
            result.append((cx, cy, cw, ch, crot, cop, self.alt, self.type))
        return result

class WeatherSystem:
    def __init__(self, seed=0):
        self.wind = EnhancedWindField()
        self.parcels = []
        self.sim_time = 0.0
        self.time_since_last_spawn = 0.0
        self.max_gap_sec = 3.0        # max allowed cloud-free interval
        
        # Global weather pattern state
        self.pattern_intensity = 0.5
        self.pattern_direction = 0.0
        self.current_formation_probability = CFG.SPAWN_PROBABILITY

    def update_weather_pattern(self):
        """Update global weather pattern that influences all clouds"""
        # Weather pattern cycles over time 
        pattern_cycle = (self.sim_time / 3600) % 24  # 24-hour cycle
        
        # Calculate pattern intensity (0-1)
        self.pattern_intensity = 0.5 + 0.5 * math.sin(pattern_cycle * math.pi / 12)
        
        # Update wind directions based on pattern
        pattern_angle = (self.sim_time / 7200) % 360  # Slowly rotating wind pattern
        self.pattern_direction = pattern_angle
        
        # Cloud formation probability varies with pattern
        base_prob = CFG.SPAWN_PROBABILITY if hasattr(CFG, 'SPAWN_PROBABILITY') else 0.04
        self.current_formation_probability = base_prob * (1 + self.pattern_intensity)

    def _spawn(self, t):
        """
        Spawn a new cloud parcel at the edge of the domain.
        
        Args:
            t: Current simulation time
        """
        # Get wind direction from wind field to determine spawn location
        _, hdg = self.wind.sample(CFG.DOMAIN_SIZE_M/2, CFG.DOMAIN_SIZE_M/2, 1000)
        d = CFG.DOMAIN_SIZE_M; b = CFG.SPAWN_BUFFER_M
        
        # Spawn at the upwind edge of the domain based on wind direction
        upwind_angle = (hdg + 180) % 360  # Opposite of wind direction
        
        # Convert angle to radians
        upwind_rad = math.radians(upwind_angle)
        
        # Calculate spawn position at domain edge in upwind direction
        edge_distance = 0.1 * d  # 10% inside domain edge
        center_x = d / 2
        center_y = d / 2
        
        # Calculate distance from center to edge in this direction
        if abs(math.cos(upwind_rad)) > 1e-6:
            # Calculate x-intercept
            if math.cos(upwind_rad) > 0:
                t_x = (d - center_x) / math.cos(upwind_rad)  # Right edge
            else:
                t_x = -center_x / math.cos(upwind_rad)  # Left edge
        else:
            t_x = float('inf')
            
        if abs(math.sin(upwind_rad)) > 1e-6:
            # Calculate y-intercept
            if math.sin(upwind_rad) > 0:
                t_y = (d - center_y) / math.sin(upwind_rad)  # Bottom edge
            else:
                t_y = -center_y / math.sin(upwind_rad)  # Top edge
        else:
            t_y = float('inf')
            
        # Use the closest intercept
        t_edge = min(t_x, t_y)
        
        # Calculate edge point
        edge_x = center_x + t_edge * math.cos(upwind_rad)
        edge_y = center_y + t_edge * math.sin(upwind_rad)
        
        # Move slightly inward from the edge
        spawn_x = edge_x - edge_distance * math.cos(upwind_rad)
        spawn_y = edge_y - edge_distance * math.sin(upwind_rad)
        
        # Add some randomization along the edge
        perpendicular_rad = upwind_rad + math.pi/2
        rand_offset = random.uniform(-0.3, 0.3) * d
        spawn_x += rand_offset * math.cos(perpendicular_rad)
        spawn_y += rand_offset * math.sin(perpendicular_rad)
        
        # Ensure spawn point is within domain bounds
        spawn_x = max(-b, min(d + b, spawn_x))
        spawn_y = max(-b, min(d + b, spawn_y))
        
        # Select cloud type based on weights
        ctype = random.choices(list(CFG.CLOUD_TYPE_WEIGHTS.keys()),
                              weights=list(CFG.CLOUD_TYPE_WEIGHTS.values()))[0]
        
        # Create new cloud parcel with enhanced visualization
        new_cloud = EnhancedCloudParcel(spawn_x, spawn_y, self.wind, ctype)
        self.parcels.append(new_cloud)
        print(f"DEBUG: Spawned new {ctype} cloud at ({spawn_x}, {spawn_y})")

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
            dt = CFG.PHYSICS_TIMESTEP
        
        # Update global weather pattern
        self.update_weather_pattern()
        
        # Update wind field
        self.wind.step(self.sim_time)
        
        # Update all parcels with wind
        expired_indices = []
        for i, parcel in enumerate(self.parcels):
            if parcel.step(dt, self.wind, self.sim_time):
                expired_indices.append(i)
        
        # Remove expired parcels (in reverse to preserve indices)
        for i in sorted(expired_indices, reverse=True):
            if i < len(self.parcels):
                print(f"DEBUG: Removing cloud that has crossed the domain")
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
        if CFG.SINGLE_CLOUD_MODE:
            if len(self.parcels) == 0:
                self._spawn(self.sim_time)
                self.time_since_last_spawn = 0.0
        else:
            # Update time since last spawn
            self.time_since_last_spawn += dt
            
            # Check if we need a forced spawn due to empty sky for too long
            need_forced = (len(self.parcels) == 0 and
                          self.time_since_last_spawn > self.max_gap_sec)
            
            # Use dynamic formation probability from weather pattern
            if len(self.parcels) < CFG.MAX_PARCELS and (
                    random.random() < self.current_formation_probability or need_forced):
                self._spawn(self.sim_time)
                self.time_since_last_spawn = 0.0
                
            # Handle forced initial cloud
            if len(self.parcels) == 0 and CFG.FORCE_INITIAL_CLOUD:
                self._spawn(self.sim_time)
                self.time_since_last_spawn = 0.0

    def get_avg_trajectory(self):
        if not self.parcels:
            return None, None, 0
        speeds = []
        directions_x = []
        directions_y = []
        for parcel in self.parcels:
            speed = math.sqrt(parcel.vx**2 + parcel.vy**2)
            direction = math.degrees(math.atan2(parcel.vy, parcel.vx)) % 360
            speeds.append(speed)
            direction_rad = math.radians(direction)
            directions_x.append(math.cos(direction_rad))
            directions_y.append(math.sin(direction_rad))
        avg_speed = sum(speeds) / len(speeds)
        avg_dir_x = sum(directions_x) / len(directions_x)
        avg_dir_y = sum(directions_y) / len(directions_y)
        avg_direction = math.degrees(math.atan2(avg_dir_y, avg_dir_x)) % 360
        mean_vector_length = math.sqrt(avg_dir_x**2 + avg_dir_y**2)
        confidence = mean_vector_length
        
        # FIXED: Speed conversion to km/h is consistent (m/s to km/h)
        avg_speed_kmh = avg_speed * 3.6  # Convert m/s to km/h 
        return avg_speed_kmh, avg_direction, confidence
        
    def current_cloud_cover_pct(self):
        total_area = 0
        domain_area = CFG.AREA_SIZE_KM * CFG.AREA_SIZE_KM
        for p in self.parcels:
            cloud_area = math.pi * (p.r ** 2)
            total_area += cloud_area
        cover = min(100, (total_area / domain_area) * 100 * 5)
        return cover

def collect_visible_ellipses(parcels):
    """Collect all visible ellipses from all parcels"""
    ellipses = []
    for parcel in parcels:
        if isinstance(parcel, EnhancedCloudParcel):
            ellipses.extend(parcel.get_ellipses())
        else:
            ellipses.append(parcel.ellipse())
    print(f"DEBUG: Collected {len(ellipses)} visible ellipses from {len(parcels)} parcels")
    return ellipses