"""
Enhanced Clear Sky Generation for Solar Farm Simulation
Calculates solar position and irradiance components
"""
import numpy as np
import math
from datetime import datetime, timedelta

class EnhancedClearSkyGeneration:
    """
    Generate realistic clear sky irradiance values.
    Uses Solar Position Algorithm (SPA) for accurate sun position.
    """
    
    def __init__(self, latitude=6.9271, longitude=79.8612, altitude=0.0):
        """
        Initialize clear sky model.
        
        Args:
            latitude: Site latitude in degrees
            longitude: Site longitude in degrees
            altitude: Site altitude in meters
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        
        # Constants
        self.solar_constant = 1367.0  # W/m²
        self.zenith_max = 87.0  # Max zenith angle for calculations
        
        # Default atmospheric parameters
        self.turbidity = 2.0
        self.water_vapor = 1.5
        self.ozone = 0.35
        self.albedo = 0.2
    
    def calculate_solar_position(self, timestamp):
        """
        Calculate solar position using SPA algorithm.
        
        Args:
            timestamp: Datetime object or hour of day
            
        Returns:
            Dictionary with 'elevation', 'azimuth', 'zenith' in degrees
        """
        # Handle both datetime and hour input
        if isinstance(timestamp, datetime):
            dt = timestamp
        else:
            # Assume it's the hour
            now = datetime.now()
            
            # FIX: Ensure hour is in 0-23 range
            hour = int(timestamp) % 24
            minute = int((timestamp % 1) * 60)
            
            # Create datetime with valid hour
            dt = datetime(now.year, now.month, now.day, hour, minute)
        
        # Calculate Julian Day
        jd = self._calculate_julian_day(dt)
        
        # Calculate solar position
        elevation, azimuth, zenith = self._calculate_spa(jd, dt)
        
        return {
            'elevation': elevation,
            'azimuth': azimuth,
            'zenith': zenith
        }
    
    def _calculate_julian_day(self, dt):
        """Calculate Julian Day from datetime"""
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        
        # Calculate time as decimal day
        day_decimal = day + hour/24.0 + minute/1440.0 + second/86400.0
        
        # Adjust month and year for January and February
        if month <= 2:
            month += 12
            year -= 1
        
        # Calculate Julian Day
        A = int(year/100)
        B = 2 - A + int(A/4)
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day_decimal + B - 1524.5
        
        return jd
    
    def _calculate_spa(self, jd, dt):
        """Simplified Solar Position Algorithm (SPA)"""
        # Time parameters
        delta_t = 67.0  # seconds, difference between UT and TT
        jde = jd + delta_t / 86400.0  # Julian Ephemeris Day
        
        # Calculate time in Julian centuries
        T = (jde - 2451545.0) / 36525.0
        
        # Geometric Mean Longitude of the Sun
        L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
        L0 = L0 % 360
        
        # Geometric Mean Anomaly of the Sun
        M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
        M = M % 360
        M_rad = math.radians(M)
        
        # Eccentricity of Earth's Orbit
        e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T**2
        
        # Equation of Center
        C = (1.914602 - 0.004817 * T - 0.000014 * T**2) * math.sin(M_rad) + \
            (0.019993 - 0.000101 * T) * math.sin(2 * M_rad) + \
            0.000289 * math.sin(3 * M_rad)
        
        # True Longitude of the Sun
        L_true = L0 + C
        
        # Apparent Longitude of the Sun (ignore aberration and nutation for simplicity)
        L_app = L_true
        
        # Obliquity of the Ecliptic
        epsilon = 23.43929 - 0.01300417 * T - 0.00000016 * T**2  # simplified
        epsilon_rad = math.radians(epsilon)
        
        # Convert to Equatorial Coordinates
        L_app_rad = math.radians(L_app)
        
        # Right Ascension
        ra_rad = math.atan2(math.cos(epsilon_rad) * math.sin(L_app_rad), math.cos(L_app_rad))
        ra = math.degrees(ra_rad)
        ra = ra % 360
        
        # Declination
        dec_rad = math.asin(math.sin(epsilon_rad) * math.sin(L_app_rad))
        dec = math.degrees(dec_rad)
        
        # Local Hour Angle
        # Convert UTC to local time (rough approximation)
        local_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        lstm = 15 * round(self.longitude / 15)  # Local Standard Time Meridian
        time_offset = 4 * (self.longitude - lstm)  # in minutes
        time_correction = time_offset / 60.0  # in hours
        local_solar_time = local_hour + time_correction
        
        # Hour angle
        hour_angle = 15.0 * (local_solar_time - 12.0)
        hour_angle_rad = math.radians(hour_angle)
        
        # Calculate zenith angle
        lat_rad = math.radians(self.latitude)
        zenith_rad = math.acos(math.sin(lat_rad) * math.sin(dec_rad) + 
                              math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_angle_rad))
        zenith = math.degrees(zenith_rad)
        
        # Calculate elevation angle
        elevation = 90.0 - zenith
        
        # Calculate azimuth angle
        azimuth_rad = math.acos((math.sin(dec_rad) - math.sin(lat_rad) * math.cos(zenith_rad)) / 
                               (math.cos(lat_rad) * math.sin(zenith_rad)))
        azimuth = math.degrees(azimuth_rad)
        
        # Adjust azimuth for afternoon hours
        if hour_angle > 0:
            azimuth = 360 - azimuth
        
        return elevation, azimuth, zenith
    
    def calculate_air_mass(self, zenith):
        """
        Calculate air mass given zenith angle.
        
        Args:
            zenith: Solar zenith angle in degrees
            
        Returns:
            Air mass ratio
        """
        # Cap zenith angle to avoid infinity
        zenith = min(zenith, self.zenith_max)
        zenith_rad = math.radians(zenith)
        
        # Kasten and Young (1989) air mass formula
        return 1.0 / (math.cos(zenith_rad) + 0.50572 * (96.07995 - zenith) ** -1.6364)
    
    def get_clear_sky_irradiance(self, timestamp):
        """
        Calculate clear sky irradiance for given timestamp.
        
        Args:
            timestamp: Datetime object or hour of day
            
        Returns:
            Dictionary with 'dni', 'dhi', 'ghi', 'poa' values in W/m²
        """
        # Get solar position
        solar_position = self.calculate_solar_position(timestamp)
        elevation = solar_position['elevation']
        zenith = solar_position['zenith']
        
        # If sun is below horizon, return zeros
        if elevation <= 0:
            return {
                'dni': 0.0,
                'dhi': 0.0,
                'ghi': 0.0,
                'poa': 0.0
            }
        
        # Calculate air mass
        air_mass = self.calculate_air_mass(zenith)
        
        # Direct Normal Irradiance (DNI)
        # Simplified Bird Clear Sky Model
        tau_r = math.exp(-0.0903 * air_mass**0.84)  # Rayleigh scattering
        tau_a = math.exp(-self.turbidity * 0.2 * air_mass**0.9)  # Aerosol extinction
        tau_w = math.exp(-0.14 * self.water_vapor * air_mass**0.3)  # Water vapor absorption
        tau_o = math.exp(-0.3 * self.ozone * air_mass**0.5)  # Ozone absorption
        
        # Transmittance product
        transmittance = tau_r * tau_a * tau_w * tau_o
        
        # Direct normal irradiance
        dni = self.solar_constant * transmittance
        
        # Diffuse horizontal irradiance (DHI)
        # Simplified model for diffuse
        dhi = self.solar_constant * 0.1 * math.sin(math.radians(elevation)) * self.turbidity
        
        # Global horizontal irradiance (GHI)
        ghi = dni * math.sin(math.radians(elevation)) + dhi
        
        # Plane of Array (POA) irradiance - assume horizontal for now
        # In a real system, this would account for module tilt and orientation
        poa = ghi
        
        return {
            'dni': dni,
            'dhi': dhi,
            'ghi': ghi,
            'poa': poa
        }
    
    def calculate_irradiance_components(self, timestamp):
        """
        Calculate irradiance components with advanced model.
        
        Args:
            timestamp: Datetime object or hour of day
            
        Returns:
            Dictionary with detailed irradiance components
        """
        # Get basic clear sky irradiance
        irradiance = self.get_clear_sky_irradiance(timestamp)
        
        # Get solar position
        solar_position = self.calculate_solar_position(timestamp)
        
        # Calculate additional components
        elevation = solar_position['elevation']
        azimuth = solar_position['azimuth']
        
        # Direct component (beam)
        dni = irradiance['dni']
        
        # Diffuse component
        dhi = irradiance['dhi']
        
        # Global horizontal
        ghi = irradiance['ghi']
        
        # Calculate incidence angle for a horizontal surface
        # In a real system, this would account for module tilt and orientation
        incidence_angle = 90.0 - elevation
        
        # Direct component on horizontal surface
        beam_horizontal = dni * math.cos(math.radians(incidence_angle))
        
        # Diffuse and reflected components
        sky_diffuse = dhi
        ground_reflected = ghi * self.albedo
        
        # Return detailed components
        return {
            'dni': dni,
            'dhi': dhi,
            'ghi': ghi,
            'poa': irradiance['poa'],
            'beam_horizontal': beam_horizontal,
            'sky_diffuse': sky_diffuse,
            'ground_reflected': ground_reflected,
            'elevation': elevation,
            'azimuth': azimuth,
            'incidence_angle': incidence_angle
        }