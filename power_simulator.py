"""
Power Simulator for Solar Farm
Realistic PV power generation with bypass diodes and Beer-Lambert light transmission
"""
import numpy as np
from datetime import datetime, timedelta
import math
from typing import Dict, List, Tuple, Optional

class PhotovoltaicCell:
    """Individual PV cell model"""
    
    def __init__(self, efficiency=0.2, area=0.0156, v_mp=0.5, i_mp=9.0):
        """
        Initialize a solar cell.
        
        Args:
            efficiency: Cell efficiency (0-1)
            area: Cell area in m²
            v_mp: Maximum power voltage in V
            i_mp: Maximum power current in A
        """
        self.efficiency = efficiency
        self.area = area  # m²
        self.v_mp = v_mp  # Voltage at maximum power point
        self.i_mp = i_mp  # Current at maximum power point
        
        # Calculate nameplate capacity
        self.power_capacity = v_mp * i_mp * efficiency
        
        # Temperature coefficient
        self.temp_coefficient = -0.004  # -0.4%/°C typical for silicon
    
    def calculate_power(self, irradiance, temperature=25.0):
        """
        Calculate power output under given conditions.
        
        Args:
            irradiance: Solar irradiance in W/m²
            temperature: Cell temperature in °C
            
        Returns:
            Dictionary with power, current, and voltage
        """
        # Temperature adjustment
        temp_factor = 1 + self.temp_coefficient * (temperature - 25)
        
        # Current is proportional to irradiance
        current = self.i_mp * (irradiance / 1000) * temp_factor
        
        # Voltage decreases slightly with increasing irradiance
        # Simplification: linear model
        voltage = self.v_mp * temp_factor
        
        # Calculate power
        power = voltage * current
        
        return {
            'power': power,
            'current': current,
            'voltage': voltage
        }

class PVModule:
    """
    PV module with bypass diodes and realistic shading behavior.
    
    A typical 60-cell module has 3 bypass diodes, each protecting 20 cells.
    When part of a string is shaded, the bypass diode activates to prevent
    the shaded cells from limiting the entire string.
    """
    
    def __init__(self, cells_per_string=20, num_strings=3, cell_efficiency=0.2):
        """
        Initialize a PV module.
        
        Args:
            cells_per_string: Number of cells per bypass diode string
            num_strings: Number of strings (bypass diodes) in module
            cell_efficiency: Efficiency of individual cells
        """
        # Create cells for each string
        self.strings = []
        for _ in range(num_strings):
            string = [PhotovoltaicCell(efficiency=cell_efficiency) for _ in range(cells_per_string)]
            self.strings.append(string)
        
        # Module parameters
        self.num_cells = cells_per_string * num_strings
        self.area = sum(cell.area for string in self.strings for cell in string)
        
        # Calculate nameplate capacity (standard test conditions)
        cell = self.strings[0][0]  # Reference cell
        self.v_mp = cell.v_mp * cells_per_string
        self.i_mp = cell.i_mp * num_strings
        self.power_capacity = self.v_mp * self.i_mp
    
    def calculate_power(self, irradiance, cell_irradiances=None, temperature=25.0):
        """
        Calculate module power with non-uniform irradiance.
        
        Args:
            irradiance: Base irradiance in W/m²
            cell_irradiances: Optional dict mapping (string_idx, cell_idx) to irradiance
            temperature: Module temperature in °C
            
        Returns:
            Dictionary with power, current, voltage, and string data
        """
        # Default to uniform irradiance if not specified
        if cell_irradiances is None:
            cell_irradiances = {}
        
        # Calculate power for each string
        string_powers = []
        string_currents = []
        string_voltages = []
        
        for s_idx, string in enumerate(self.strings):
            # Find minimum irradiance in the string (limiting factor)
            string_irradiances = []
            for c_idx, cell in enumerate(string):
                # Get specific cell irradiance or use base
                cell_key = (s_idx, c_idx)
                if cell_key in cell_irradiances:
                    string_irradiances.append(cell_irradiances[cell_key])
                else:
                    string_irradiances.append(irradiance)
            
            # Check if bypass diode activates
            if min(string_irradiances) < 100:  # Bypass activates if any cell < 100 W/m²
                # Bypass diode activated, string contributes no voltage but passes current
                string_power = 0
                string_current = self.i_mp * (irradiance / 1000)  # Use base irradiance for current
                string_voltage = 0
            else:
                # String operates at its minimum irradiance
                min_irradiance = min(string_irradiances)
                
                # Sum up all cell powers
                total_power = 0
                for cell in string:
                    cell_output = cell.calculate_power(min_irradiance, temperature)
                    total_power += cell_output['power']
                
                # String operates as a unit
                string_power = total_power
                string_current = string[0].calculate_power(min_irradiance, temperature)['current']
                string_voltage = self.v_mp / len(self.strings)  # Simplified
            
            string_powers.append(string_power)
            string_currents.append(string_current)
            string_voltages.append(string_voltage)
        
        # Total module power is sum of string powers
        total_power = sum(string_powers)
        
        # Current is the sum of string currents
        total_current = sum(string_currents)
        
        # Voltage is the average of string voltages
        total_voltage = sum(string_voltages)
        
        return {
            'power': total_power,
            'current': total_current,
            'voltage': total_voltage,
            'string_powers': string_powers,
            'string_currents': string_currents,
            'string_voltages': string_voltages,
            'bypass_activated': [p == 0 for p in string_powers]
        }
    
    def simulate_partial_shading(self, base_irradiance, shading_pattern):
        """
        Simulate effect of partial shading.
        
        Args:
            base_irradiance: Clear sky irradiance in W/m²
            shading_pattern: List of tuples (string_idx, cell_idx, shade_factor)
            
        Returns:
            Power output dictionary
        """
        # Create cell irradiances dict
        cell_irradiances = {}
        
        for s_idx, c_idx, shade_factor in shading_pattern:
            # Calculate attenuated irradiance
            cell_irradiances[(s_idx, c_idx)] = base_irradiance * (1 - shade_factor)
        
        # Calculate power with this shading pattern
        return self.calculate_power(base_irradiance, cell_irradiances)
    
    def apply_shadow_coverage(self, direct_irradiance, diffuse_irradiance, shadow_coverage):
        """
        Apply shadow coverage to the module with separate direct and diffuse components.
        
        Args:
            direct_irradiance: Direct component of irradiance in W/m²
            diffuse_irradiance: Diffuse component of irradiance in W/m²
            shadow_coverage: Shadow coverage from 0 (no shadow) to 1 (full shadow)
            
        Returns:
            Power output dictionary
        """
        # For simplified model, assume first string gets shaded first
        shading_pattern = []
        
        # No shading case
        if shadow_coverage <= 0:
            total_irradiance = direct_irradiance + diffuse_irradiance
            return self.calculate_power(total_irradiance)
        
        # Shade affects direct component based on coverage
        attenuated_direct = direct_irradiance * (1 - shadow_coverage)
        
        # Diffuse component is only reduced if cloud cover is high
        diffuse_factor = 1.0 if shadow_coverage < 0.5 else 1.0 - (shadow_coverage - 0.5) * 2
        attenuated_diffuse = diffuse_irradiance * diffuse_factor
        
        # Total irradiance for each cell
        total_irradiance = attenuated_direct + attenuated_diffuse
        
        # Full shading case
        if shadow_coverage >= 0.95:
            # Shade all cells
            for s_idx in range(len(self.strings)):
                for c_idx in range(len(self.strings[s_idx])):
                    shading_pattern.append((s_idx, c_idx, shadow_coverage))
            
            return self.simulate_partial_shading(direct_irradiance + diffuse_irradiance, shading_pattern)
        
        # Partial shading: shade cells proportionally
        num_strings = len(self.strings)
        cells_per_string = len(self.strings[0])
        total_cells = num_strings * cells_per_string
        
        # Calculate how many cells to shade
        cells_to_shade = int(shadow_coverage * total_cells)
        
        # Shade cells sequentially across strings
        for i in range(cells_to_shade):
            s_idx = i // cells_per_string
            c_idx = i % cells_per_string
            shading_pattern.append((s_idx, c_idx, shadow_coverage))
        
        return self.simulate_partial_shading(direct_irradiance + diffuse_irradiance, shading_pattern)

class OpticalModel:
    """Calculate optical properties of atmosphere and clouds"""
    
    def __init__(self):
        """Initialize optical model"""
        pass
    
    def calculate_clear_sky_irradiance(self, solar_elevation, turbidity=2.0):
        """
        Calculate clear sky irradiance components.
        
        Args:
            solar_elevation: Solar elevation angle in degrees
            turbidity: Linke turbidity factor (2-5 typical)
            
        Returns:
            Dictionary with DNI, DHI, and GHI components in W/m²
        """
        if solar_elevation <= 0:
            return {'dni': 0, 'dhi': 0, 'ghi': 0}
        
        # Simplified clear sky model
        # Extraterrestrial radiation
        I0 = 1367  # W/m²
        
        # Air mass
        AM = 1 / np.sin(np.radians(solar_elevation))
        
        # Direct normal irradiance (DNI)
        dni = I0 * 0.7**(AM**0.678) * 0.14**(turbidity - 1)
        
        # Diffuse horizontal irradiance (DHI)
        dhi = I0 * 0.1 * np.sin(np.radians(solar_elevation)) * turbidity
        
        # Global horizontal irradiance (GHI)
        ghi = dni * np.sin(np.radians(solar_elevation)) + dhi
        
        return {
            'dni': dni,
            'dhi': dhi,
            'ghi': ghi
        }
    
    def calculate_cloud_optical_depth(self, opacity, cloud_type='cumulus'):
        """
        Calculate cloud optical depth based on visual opacity.
        
        Args:
            opacity: Visual opacity from 0 (clear) to 1 (opaque)
            cloud_type: Type of cloud
            
        Returns:
            Optical depth
        """
        # Base optical depth ranges
        optical_ranges = {
            'cirrus': (0, 4),
            'altocumulus': (4, 10),
            'cumulus': (10, 25),
            'stratus': (15, 30),
            'cumulonimbus': (20, 100)
        }
        
        # Get range for this cloud type
        tau_min, tau_max = optical_ranges.get(cloud_type, (10, 25))
        
        # Non-linear mapping (lower opacities have less effect)
        if opacity <= 0:
            return 0
        
        # Power function for better perceptual scaling
        # Quadratic relationship matches visual perception better
        normalized_tau = opacity**2  
        
        # Scale to range
        tau = tau_min + normalized_tau * (tau_max - tau_min)
        
        return tau
    
    def calculate_beer_lambert_transmission(self, optical_depth, solar_elevation):
        """
        Calculate light transmission using Beer-Lambert law.
        
        Args:
            optical_depth: Cloud optical depth
            solar_elevation: Solar elevation in degrees
            
        Returns:
            Transmission factor (0-1)
        """
        # Prevent division by zero
        sin_elevation = max(0.01, np.sin(np.radians(solar_elevation)))
        
        # Beer-Lambert law: I = I0 * exp(-tau / sin(theta))
        transmission = np.exp(-optical_depth / sin_elevation)
        
        return transmission
    
    def calculate_diffuse_enhancement(self, optical_depth):
        """
        Calculate diffuse light enhancement from clouds.
        
        Args:
            optical_depth: Cloud optical depth
            
        Returns:
            Diffuse enhancement factor
        """
        # Thin clouds increase diffuse, thick clouds block it
        if optical_depth < 5:
            # Thin clouds enhance diffuse light
            enhancement = 1.0 + 0.2 * optical_depth  # Up to 2x for thin clouds
        else:
            # Thicker clouds start to reduce diffuse light
            enhancement = 2.0 * np.exp(-0.1 * (optical_depth - 5))
        
        return enhancement
    
    def calculate_irradiance_components(self, solar_position, cloud_opacity=0, cloud_type='cumulus'):
        """
        Calculate irradiance components with cloud effects.
        
        Args:
            solar_position: Dict with 'elevation' and 'azimuth' in degrees
            cloud_opacity: Cloud opacity (0-1)
            cloud_type: Type of cloud
            
        Returns:
            Dictionary with irradiance components in W/m²
        """
        # Extract solar elevation
        solar_elevation = solar_position['elevation']
        
        # Get clear sky irradiance
        clear_sky = self.calculate_clear_sky_irradiance(solar_elevation)
        
        # If no sun, return zero values
        if solar_elevation <= 0:
            return {
                'dni': 0,
                'dhi': 0,
                'ghi': 0,
                'clear_sky_dni': 0,
                'clear_sky_dhi': 0,
                'clear_sky_ghi': 0,
                'transmission': 0,
                'diffuse_factor': 0,
                'direct': 0,
                'diffuse': 0,
                'total': 0
            }
        
        # If no clouds, return clear sky values with direct/diffuse components
        if cloud_opacity <= 0:
            # Calculate incidence angle for horizontal surface
            # For tilted modules, this would use the actual tilt and orientation
            incidence_angle = 90 - solar_elevation
            
            # Direct component
            direct = clear_sky['dni'] * np.cos(np.radians(incidence_angle))
            
            # Diffuse component (simplified Hay-Davies model)
            diffuse = 0.2 * clear_sky['ghi']
            
            # Total irradiance
            total = direct + diffuse
            
            return {
                'dni': clear_sky['dni'],
                'dhi': clear_sky['dhi'],
                'ghi': clear_sky['ghi'],
                'clear_sky_dni': clear_sky['dni'],
                'clear_sky_dhi': clear_sky['dhi'],
                'clear_sky_ghi': clear_sky['ghi'],
                'transmission': 1.0,
                'diffuse_factor': 1.0,
                'direct': direct,
                'diffuse': diffuse,
                'total': total
            }
        
        # Calculate cloud optical depth
        optical_depth = self.calculate_cloud_optical_depth(cloud_opacity, cloud_type)
        
        # Calculate direct transmission (Beer-Lambert)
        direct_transmission = self.calculate_beer_lambert_transmission(optical_depth, solar_elevation)
        
        # Calculate diffuse enhancement
        diffuse_factor = self.calculate_diffuse_enhancement(optical_depth)
        
        # Calculate components with cloud effects
        dni = clear_sky['dni'] * direct_transmission
        dhi = clear_sky['dhi'] * diffuse_factor
        ghi = dni * np.sin(np.radians(solar_elevation)) + dhi
        
        # Calculate incidence angle for horizontal surface
        incidence_angle = 90 - solar_elevation
        
        # Direct component
        direct = dni * np.cos(np.radians(incidence_angle))
        
        # Diffuse component (simplified Hay-Davies model)
        diffuse = 0.2 * ghi
        
        # Ensure diffuse is preserved unless cloud cover is high
        if cloud_opacity < 0.5:
            diffuse = 0.2 * clear_sky['ghi']  # Keep original diffuse
        else:
            diffuse = 0.2 * clear_sky['ghi'] * (1.0 - (cloud_opacity - 0.5) * 2)
        
        # Total irradiance
        total = direct + diffuse
        
        return {
            'dni': dni,
            'dhi': dhi,
            'ghi': ghi,
            'clear_sky_dni': clear_sky['dni'],
            'clear_sky_dhi': clear_sky['dhi'],
            'clear_sky_ghi': clear_sky['ghi'],
            'transmission': direct_transmission,
            'diffuse_factor': diffuse_factor,
            'direct': direct,
            'diffuse': diffuse,
            'total': total
        }

class PowerSimulator:
    """
    Simulate solar power generation for the farm.
    """
    
    def __init__(self, panel_df=None, latitude=6.9271, longitude=79.8612):
        """
        Initialize power simulator.
        
        Args:
            panel_df: DataFrame with panel information
            latitude: Site latitude
            longitude: Site longitude
        """
        self.panel_df = panel_df
        self.latitude = latitude
        self.longitude = longitude
        
        # Solar times
        self.sunrise_hour = 6.5  # 6:30 AM
        self.sunset_hour = 18.5  # 6:30 PM
        
        # Create modules for each panel
        self.modules = {}
        if panel_df is not None:
            for _, row in panel_df.iterrows():
                panel_id = row["panel_id"]
                # Create a standard 60-cell module with 3 bypass diodes
                self.modules[panel_id] = PVModule(cells_per_string=20, num_strings=3)
        
        # Create optical model
        self.optical_model = OpticalModel()
        
        # Create clear sky model
        try:
            from clear_sky_generation import EnhancedClearSkyGeneration
            self.clear_sky_model = EnhancedClearSkyGeneration(
                latitude=latitude,
                longitude=longitude
            )
        except ImportError:
            print("Warning: EnhancedClearSkyGeneration not available, using simplified model")
            self.clear_sky_model = self._create_simplified_model()
    
    def _create_simplified_model(self):
        """Create a simplified clear sky model if the enhanced model is not available"""
        class SimplifiedClearSkyModel:
            def __init__(self, latitude, longitude):
                self.latitude = latitude
                self.longitude = longitude
            
            def calculate_solar_position(self, hour):
                # Simple solar position calculation
                if hour < 6 or hour > 18:
                    return {'elevation': -10, 'azimuth': 0, 'zenith': 100}
                
                # Approximate solar path
                norm_hour = (hour - 6) / 12  # 0 at sunrise, 1 at sunset
                elevation = 70 * math.sin(math.pi * norm_hour)
                azimuth = 180 * norm_hour
                
                return {
                    'elevation': elevation,
                    'azimuth': azimuth,
                    'zenith': 90 - elevation
                }
            
            def get_clear_sky_irradiance(self, hour):
                solar_pos = self.calculate_solar_position(hour)
                
                if solar_pos['elevation'] <= 0:
                    return {'dni': 0, 'dhi': 0, 'ghi': 0, 'poa': 0}
                
                # Simple approximation
                max_dni = 1000  # W/m²
                max_dhi = 100   # W/m²
                
                # Scale with elevation angle
                norm_factor = solar_pos['elevation'] / 90
                dni = max_dni * norm_factor
                dhi = max_dhi * norm_factor
                ghi = dni * math.sin(math.radians(solar_pos['elevation'])) + dhi
                
                return {
                    'dni': dni,
                    'dhi': dhi,
                    'ghi': ghi,
                    'poa': ghi
                }
            
            def calculate_irradiance_components(self, hour):
                solar_pos = self.calculate_solar_position(hour)
                irradiance = self.get_clear_sky_irradiance(hour)
                
                if solar_pos['elevation'] <= 0:
                    return {
                        'dni': 0, 'dhi': 0, 'ghi': 0, 'poa': 0,
                        'beam_horizontal': 0, 'sky_diffuse': 0, 'ground_reflected': 0,
                        'elevation': solar_pos['elevation'], 'azimuth': solar_pos['azimuth'],
                        'incidence_angle': 90
                    }
                
                # Calculate components
                beam_horizontal = irradiance['dni'] * math.sin(math.radians(solar_pos['elevation']))
                sky_diffuse = irradiance['dhi']
                ground_reflected = irradiance['ghi'] * 0.2  # Assuming albedo of 0.2
                
                return {
                    'dni': irradiance['dni'],
                    'dhi': irradiance['dhi'],
                    'ghi': irradiance['ghi'],
                    'poa': irradiance['poa'],
                    'beam_horizontal': beam_horizontal,
                    'sky_diffuse': sky_diffuse,
                    'ground_reflected': ground_reflected,
                    'elevation': solar_pos['elevation'],
                    'azimuth': solar_pos['azimuth'],
                    'incidence_angle': 90 - solar_pos['elevation']
                }
        
        return SimplifiedClearSkyModel(self.latitude, self.longitude)
    
    def calculate_solar_position(self, timestamp):
        """
        Calculate solar position for given time.
        
        Args:
            timestamp: Datetime or hour of day
            
        Returns:
            Dictionary with 'elevation' and 'azimuth' in degrees
        """
        # Use the enhanced clear sky model for accurate solar position
        return self.clear_sky_model.calculate_solar_position(timestamp)
    
    def calculate_power(self, hour, panel_coverage):
        """
        Calculate power output for each panel.
        
        Args:
            hour: Hour of day
            panel_coverage: Dictionary mapping panel_id to shadow coverage
            
        Returns:
            Dictionary mapping panel_id to power output
        """
        # Ensure we have valid panel_df
        if self.panel_df is None or len(self.panel_df) == 0:
            return {'total': 0.0, 'baseline_total': 0.0, 'farm_reduction_pct': 0.0}
        
        # Get solar position
        solar_position = self.calculate_solar_position(hour)
        
        # If sun is below horizon, no power
        if solar_position['elevation'] <= 0:
            total_power = 0.0
            power_output = {
                'total': total_power,
                'baseline_total': 0.0,
                'farm_reduction_pct': 0.0
            }
            
            # Add zero power for each panel
            if self.panel_df is not None:
                for _, row in self.panel_df.iterrows():
                    panel_id = row["panel_id"]
                    power_output[panel_id] = {
                        'baseline': 0,
                        'final_power': 0,
                        'coverage': 0,
                        'direct': 0,
                        'diffuse': 0
                    }
            
            return power_output
        
        # Get clear sky irradiance from enhanced model
        try:
            clear_sky = self.clear_sky_model.get_clear_sky_irradiance(hour)
        except:
            # Fallback in case of error
            clear_sky = {'dni': 1000, 'dhi': 100, 'ghi': 1000}
        
        # Calculate power for each panel
        power_output = {}
        total_power = 0.0
        
        for _, row in self.panel_df.iterrows():
            panel_id = row["panel_id"]
            
            # Get panel parameters
            if "power_capacity" in row:
                capacity = row["power_capacity"]
            else:
                capacity = 5.0  # Default capacity in kW
            
            # Get shadow coverage
            coverage = panel_coverage.get(panel_id, 0.0)
            
            # Calculate irradiance components with cloud effects
            irradiance = self.optical_model.calculate_irradiance_components(
                solar_position, 
                cloud_opacity=coverage, 
                cloud_type='cumulus'
            )
            
            # Extract irradiance values
            direct = irradiance['direct']
            diffuse = irradiance['diffuse']
            total_irradiance = irradiance['total']
            
            # Reference clear sky values
            clear_direct = clear_sky['dni'] * np.cos(np.radians(90 - solar_position['elevation']))
            clear_diffuse = 0.2 * clear_sky['ghi']
            clear_total = clear_direct + clear_diffuse
            
            # Get module power with bypass diode effects
            module = self.modules.get(panel_id)
            if module:
                # Apply realistic PV module model with separate direct/diffuse components
                module_result = module.apply_shadow_coverage(direct, diffuse, coverage)
                power = module_result['power'] / 1000  # Convert W to kW
                
                # Calculate power reduction factors
                transmission = irradiance.get('transmission', 1.0)
                diffuse_factor = irradiance.get('diffuse_factor', 1.0)
                
                # Store detailed module info
                panel_info = {
                    'baseline': capacity,
                    'final_power': power,
                    'coverage': coverage,
                    'transmission': transmission,
                    'diffuse_factor': diffuse_factor,
                    'direct': direct,
                    'diffuse': diffuse,
                    'total_irradiance': total_irradiance,
                    'clear_irradiance': clear_total,
                    'bypass_activated': module_result.get('bypass_activated', [])
                }
            else:
                # Simple model (just scale by irradiance ratio)
                irradiance_ratio = total_irradiance / clear_total if clear_total > 0 else 0
                power = capacity * irradiance_ratio
                
                # Store simplified info
                panel_info = {
                    'baseline': capacity,
                    'final_power': power,
                    'coverage': coverage,
                    'direct': direct,
                    'diffuse': diffuse,
                    'total_irradiance': total_irradiance,
                    'clear_irradiance': clear_total,
                    'irradiance_ratio': irradiance_ratio
                }
            
            # Add to total
            total_power += power
            
            # Store panel power info
            power_output[panel_id] = panel_info
        
        # Add total power
        power_output['total'] = total_power
        
        # Add baseline (clear sky) total
        baseline_total = sum(info.get('baseline', 0) for panel_id, info in power_output.items() 
                           if panel_id != 'total' and panel_id != 'baseline_total' and panel_id != 'farm_reduction_pct')
        power_output['baseline_total'] = baseline_total
        
        # Calculate farm-wide reduction percentage
        if baseline_total > 0:
            power_output['farm_reduction_pct'] = (baseline_total - total_power) / baseline_total * 100
        else:
            power_output['farm_reduction_pct'] = 0.0
        
        return power_output