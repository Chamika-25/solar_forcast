"""
Shadow Calculator for Solar Farm Simulation
Calculates cloud shadow coverage for solar panels and its effect on power generation
"""
import numpy as np
import math
from collections import defaultdict, deque, namedtuple
import time

# Define a standardized structure for cloud ellipses
CloudEllipse = namedtuple('CloudEllipse', ['cx', 'cy', 'width', 'height', 'rotation', 'opacity', 'altitude'])

# Helper function to convert from tuple to CloudEllipse
def to_cloud_ellipse(ellipse_params):
    """Convert tuple to CloudEllipse with proper default for altitude"""
    if len(ellipse_params) >= 7:
        return CloudEllipse(*ellipse_params)
    else:
        cx, cy, cw, ch, crot, cop = ellipse_params
        return CloudEllipse(cx, cy, cw, ch, crot, cop, 1.0)

class ShadowCalculator:
    """Calculate shadow coverage from cloud positions and shapes"""
    
    def __init__(self, domain_size=50000, area_size_km=50.0):
        """
        Initialize the shadow calculator.
        
        Args:
            domain_size: Size of the simulation domain in pixels
            area_size_km: Size of the simulation area in kilometers
        """
        self.domain_size = domain_size
        self.area_size_km = area_size_km
        self.scale_factor = domain_size / area_size_km
        
        # Shadow settings
        self.cloud_transmittance = 0.2  # 0 = complete blackout, 1 = no effect
        self.shadow_fade_ms = 500  # milliseconds for shadow to fully apply/remove
        self.penumbra_width = 60 / 1000  # penumbra width in km (60m)
        
        # Shadow history for smooth transitions
        self.shadow_history = {}
        self.last_update_time = time.time() * 1000  # ms
        
        # Spatial indexing for performance
        self.spatial_cell_size = 2.0  # km
        
        # Cloud-specific spatial index (rebuilt each frame)
        self.cloud_cells = defaultdict(list)
        
        # Solar position cache
        self.last_solar_position = None
    
    def configure(self, settings):
        """Configure from settings object"""
        if hasattr(settings, 'CLOUD_TRANSMITTANCE'):
            self.cloud_transmittance = settings.CLOUD_TRANSMITTANCE
        if hasattr(settings, 'SHADOW_FADE_MS'):
            self.shadow_fade_ms = settings.SHADOW_FADE_MS
        if hasattr(settings, 'PENUMBRA_WIDTH'):
            self.penumbra_width = settings.PENUMBRA_WIDTH / 1000  # convert m to km
    
    def calculate_panel_coverage(self, cloud_ellipses, panel_df, solar_position=None, panel_cells=None):
        """
        Calculate shadow coverage for each panel considering sun position.
        
        Args:
            cloud_ellipses: List of cloud ellipse parameters
            panel_df: DataFrame with panel information
            solar_position: Dict with 'elevation' and 'azimuth' in degrees
            panel_cells: Dictionary mapping cell coordinates to panel IDs
        
        Returns:
            Dictionary mapping panel_id to coverage (0-1)
        """
        # Skip calculation if no clouds
        if not cloud_ellipses:
            return {}
        
        # Default solar position (vertical sun) if not provided
        if solar_position is None:
            solar_position = {'elevation': 90.0, 'azimuth': 0.0}
        
        # Cache solar position
        self.last_solar_position = solar_position
        
        # Clear cloud cells and rebuild for this frame
        self.cloud_cells.clear()
        
        # Project clouds to ground based on solar position
        projected_clouds = self._project_clouds_to_ground(cloud_ellipses, solar_position)
        
        # Calculate panel coverage
        coverage_dict = {}
        current_time = time.time() * 1000  # ms
        elapsed_ms = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Process panel coverage efficiently using spatial cells
        for _, row in panel_df.iterrows():
            panel_id = row["panel_id"]
            panel_x_km = row["x_km"]
            panel_y_km = row["y_km"]
            
            # Get cell for this panel
            cell_x = int(panel_x_km / self.spatial_cell_size)
            cell_y = int(panel_y_km / self.spatial_cell_size)
            cell_key = (cell_x, cell_y)
            
            # Only check clouds that could overlap this cell
            relevant_clouds = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (cell_x + dx, cell_y + dy)
                    if neighbor_key in self.cloud_cells:
                        cloud_indices = self.cloud_cells[neighbor_key]
                        # Only add integer indices (defensive programming)
                        relevant_clouds.extend([idx for idx in cloud_indices if isinstance(idx, int)])
            
            # Get maximum coverage from all relevant clouds
            max_coverage = 0
            
            for cloud_idx in set(relevant_clouds):
                if cloud_idx >= len(projected_clouds):
                    continue
                    
                cloud = projected_clouds[cloud_idx]
                coverage = self._calculate_point_coverage(
                    panel_x_km, panel_y_km, cloud
                )
                max_coverage = max(max_coverage, coverage)
            
            # Apply smooth transitions if configured
            if self.shadow_fade_ms > 0:
                # Get previous coverage and update
                prev_coverage = self.shadow_history.get(panel_id, 0.0)
                
                # Calculate step size for this frame
                max_step = elapsed_ms / self.shadow_fade_ms
                
                # Smooth transition
                if max_coverage > prev_coverage:
                    # Shadow increasing
                    coverage = min(max_coverage, prev_coverage + max_step)
                else:
                    # Shadow decreasing
                    coverage = max(max_coverage, prev_coverage - max_step)
                
                # Update history
                self.shadow_history[panel_id] = coverage
            else:
                # No smoothing
                coverage = max_coverage
            
            # Only include panels with coverage > 0
            if coverage > 0:
                coverage_dict[panel_id] = coverage
        
        return coverage_dict
    
    def _project_clouds_to_ground(self, cloud_ellipses, solar_position):
        """
        Project cloud ellipses to ground based on solar position.
        
        Args:
            cloud_ellipses: List of cloud ellipse parameters
            solar_position: Dict with 'elevation' and 'azimuth' in degrees
        
        Returns:
            List of projected clouds with their parameters
        """
        # Convert solar position to radians
        elevation_rad = math.radians(solar_position['elevation'])
        azimuth_rad = math.radians(solar_position['azimuth'])
        
        # Avoid division by zero with very low sun angles
        elevation_rad = max(elevation_rad, math.radians(1.0))
        
        projected_clouds = []
        # Completely clear the cloud cells for this frame
        self.cloud_cells = defaultdict(list)
        
        for i, ellipse_params in enumerate(cloud_ellipses):
            # Handle both 6 and 7-element tuples
            if len(ellipse_params) >= 7:
                cx, cy, cw, ch, crot, cop, cloud_altitude_km = ellipse_params
            else:
                cx, cy, cw, ch, crot, cop = ellipse_params
                # Assume a default cloud altitude (higher for bigger clouds)
                cloud_altitude_km = 1.0 + (cw / self.scale_factor / 10.0)  # Simple heuristic
            
            # Skip fully transparent clouds
            if cop < 0.05:
                continue
                
            # Convert domain coordinates to km
            cx_km = cx / self.scale_factor
            cy_km = cy / self.scale_factor
            cw_km = cw / self.scale_factor
            ch_km = ch / self.scale_factor
            
            # Calculate shadow projection shift based on solar position
            # For sun not directly overhead, shadows are offset
            zenith_rad = math.pi/2 - elevation_rad
            shadow_dx = cloud_altitude_km * math.tan(zenith_rad) * math.cos(azimuth_rad)
            shadow_dy = cloud_altitude_km * math.tan(zenith_rad) * math.sin(azimuth_rad)
            
            # Calculate shadow stretch/shear based on solar elevation
            # Low sun angles create longer shadows
            stretch_factor = 1.0 / math.sin(elevation_rad) if elevation_rad > 0 else 10.0
            
            # For simplicity, we'll just stretch the ellipse in the azimuth direction
            new_width = cw_km * stretch_factor if stretch_factor > 1.0 else cw_km
            new_height = ch_km
            
            # Adjust rotation based on solar azimuth
            new_rotation = crot + azimuth_rad
            
            # Create projected cloud
            projected_cloud = {
                'x': cx_km + shadow_dx,
                'y': cy_km + shadow_dy,
                'width': new_width,
                'height': new_height,
                'rotation': new_rotation,
                'opacity': cop,
                'altitude': cloud_altitude_km
            }
            
            projected_clouds.append(projected_cloud)
            
            # Update spatial index with cloud cells
            # Use a bounding box for the cloud
            cloud_radius_km = max(new_width, new_height) / 2
            min_x = int((cx_km + shadow_dx - cloud_radius_km) / self.spatial_cell_size)
            max_x = int((cx_km + shadow_dx + cloud_radius_km) / self.spatial_cell_size) + 1
            min_y = int((cy_km + shadow_dy - cloud_radius_km) / self.spatial_cell_size)
            max_y = int((cy_km + shadow_dy + cloud_radius_km) / self.spatial_cell_size) + 1
            
            for cell_x in range(min_x, max_x):
                for cell_y in range(min_y, max_y):
                    # Store only the integer index
                    self.cloud_cells[(cell_x, cell_y)].append(i)
        
        return projected_clouds
    
    def _calculate_point_coverage(self, px, py, cloud):
        """
        Calculate coverage for a point (panel) from a projected cloud.
        
        Args:
            px, py: Point coordinates in km
            cloud: Projected cloud parameters
        
        Returns:
            Coverage value between 0 and 1
        """
        # Extract cloud parameters
        cx = cloud['x']
        cy = cloud['y']
        width = cloud['width']
        height = cloud['height']
        rotation = cloud['rotation']
        opacity = cloud['opacity']
        cloud_altitude = cloud.get('altitude', 1.0)  # Default 1 km if not specified
        
        # Fast bounding box check first
        cloud_radius = max(width, height) / 2
        dx = px - cx
        dy = py - cy
        distance_sq = dx*dx + dy*dy
        
        if distance_sq > cloud_radius*cloud_radius:
            return 0.0
        
        # Calculate distance to ellipse edge considering rotation
        # Translate point to ellipse-centered coordinate system
        tx = px - cx
        ty = py - cy
        
        # Rotate point to align with ellipse axes
        rx = tx * math.cos(-rotation) - ty * math.sin(-rotation)
        ry = tx * math.sin(-rotation) + ty * math.cos(-rotation)
        
        # Calculate normalized distance from center
        nx = (2 * rx / width)
        ny = (2 * ry / height)
        distance_normalized = math.sqrt(nx*nx + ny*ny)
        
        # Calculate penumbra width based on cloud altitude
        # Penumbra width = 0.2 × altitude (km) in ground km
        altitude_based_penumbra = 0.2 * cloud_altitude
        penumbra_width_km = max(self.penumbra_width, altitude_based_penumbra)
        
        # Apply penumbra effect
        if distance_normalized <= 1.0:
            # Inside ellipse - full shadow
            penumbra_factor = 1.0
        elif distance_normalized <= 1.0 + (penumbra_width_km / cloud_radius):
            # Inside penumbra - linear falloff
            penumbra_factor = 1.0 - (distance_normalized - 1.0) / (penumbra_width_km / cloud_radius)
        else:
            # Outside shadow
            penumbra_factor = 0.0
        
        # Final coverage is opacity * penumbra factor
        return opacity * penumbra_factor
    
    def calculate_power_reduction(self, panel_coverage, panel_df, baseline_power=None, solar_position=None):
        """
        Calculate power reduction for each panel based on shadow coverage.
        
        Args:
            panel_coverage: Dictionary mapping panel_id to coverage (0-1)
            panel_df: DataFrame with panel information
            baseline_power: Optional dictionary of baseline power values
            solar_position: Dict with solar position info (for Beer-Lambert)
        
        Returns:
            Dictionary mapping panel_id to power output information
        """
        if solar_position is None and self.last_solar_position is not None:
            solar_position = self.last_solar_position
        
        # Default solar elevation for Beer-Lambert
        solar_elevation = 90.0 if solar_position is None else solar_position.get('elevation', 90.0)
        
        power_output = {}
        total_power = 0.0
        total_baseline = 0.0
        
        for _, row in panel_df.iterrows():
            panel_id = row["panel_id"]
            
            # Get panel capacity
            if "power_capacity" in row:
                capacity = row["power_capacity"]
            else:
                capacity = 5.0  # Default capacity in kW
            
            # Get baseline power (could be time-of-day dependent)
            if baseline_power is not None and panel_id in baseline_power:
                baseline = baseline_power[panel_id]
            else:
                baseline = capacity
            
            # Calculate power reduction from shadow
            coverage = panel_coverage.get(panel_id, 0.0)
            
            # Beer-Lambert law for light transmission
            # I = I0 * exp(-tau / sin(elevation))
            sin_elevation = math.sin(math.radians(solar_elevation))
            if sin_elevation < 0.01:  # Prevent division by zero
                sin_elevation = 0.01
                
            # Calculate optical depth based on coverage
            # Simple model: tau proportional to coverage
            optical_depth = coverage * 5.0  # Scale factor for reasonable attenuation
            
            # Calculate direct component (Beer-Lambert)
            transmittance_direct = math.exp(-optical_depth / sin_elevation)
            
            # Add diffuse component (even with high optical depth, there's some diffuse light)
            diffuse_fraction = 0.2  # 20% diffuse under clear skies
            diffuse_attenuation = 0.5  # Diffuse light attenuated less by clouds
            transmittance_diffuse = diffuse_fraction * math.exp(-optical_depth * diffuse_attenuation)
            
            # Total effective transmittance
            effective_transmittance = transmittance_direct * (1.0 - diffuse_fraction) + transmittance_diffuse
            
            # Apply minimum transmittance (even in full shadow, some light gets through)
            effective_transmittance = max(effective_transmittance, self.cloud_transmittance)
            
            # Calculate power
            power = baseline * effective_transmittance
            
            # Store power output
            power_output[panel_id] = {
                'baseline': baseline,
                'coverage': coverage,
                'optical_depth': optical_depth,
                'transmittance': effective_transmittance,
                'final_power': power
            }
            
            # Update totals
            total_power += power
            total_baseline += baseline
        
        # Add totals to output
        power_output['total'] = total_power
        power_output['baseline_total'] = total_baseline
        power_output['farm_reduction_pct'] = (
            (total_baseline - total_power) / total_baseline * 100 
            if total_baseline > 0 else 0.0
        )
        
        return power_output
    
    def get_coverage_stats(self, panel_coverage, panel_df):
        """
        Calculate coverage statistics.
        
        Args:
            panel_coverage: Dictionary mapping panel_id to coverage (0-1)
            panel_df: DataFrame with panel information
        
        Returns:
            Dictionary with statistics
        """
        if not panel_coverage:
            return {
                'affected_count': 0,
                'total_count': len(panel_df),
                'affected_pct': 0.0,
                'avg_coverage': 0.0,
                'max_coverage': 0.0
            }
        
        affected_count = len(panel_coverage)
        total_count = len(panel_df)
        affected_pct = affected_count / total_count * 100 if total_count > 0 else 0
        
        coverages = list(panel_coverage.values())
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0
        max_coverage = max(coverages) if coverages else 0
        
        return {
            'affected_count': affected_count,
            'total_count': total_count,
            'affected_pct': affected_pct,
            'avg_coverage': avg_coverage,
            'max_coverage': max_coverage
        }