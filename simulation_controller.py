import numpy as np
from datetime import datetime, timedelta
import time
import pygame
import math
import sim_config as CFG
import os
import sys
from panel_layout import PANELS, build_panel_cells, panel_df

# Add the renderer path to the system path
sys.path.append('C:/Users/Suchira_Garusinghe/Desktop/Simulation/simulation8.5.0/pygame_rendereres')

class SimulationController:
   def __init__(self, start_time=11.0, duration_hours=None, timestep_minutes=5, real_time_output=False, debug_mode=False, panel_df=None):
       self.start_time = start_time
       self.duration_hours = duration_hours
       self.timestep_minutes = timestep_minutes
       self.real_time_output = real_time_output
       self.debug_mode = debug_mode
       self.current_hour = start_time
       self.panel_df = panel_df if panel_df is not None else panel_df
       self.CELL_SIZE_KM = 2.0
       self.panel_cells = build_panel_cells(self.CELL_SIZE_KM)
       self._initialize_systems()
       self.frame_count = 0
       self.last_physics_time = time.time()
       self.accumulated_dt = 0.0
       self.font = None
       self.large_font = None
       self.info_panel_bg = None
       
   def _initialize_systems(self):
       try:
           from cloud_simulation import WeatherSystem, collect_visible_ellipses
           from pygame_rendereres.cloud_renderer import create_cloud_surface
           from shadow_calculator import ShadowCalculator
           self.weather_system = WeatherSystem()
           print("Weather system initialized")
           self.shadow_calculator = ShadowCalculator(
               domain_size=CFG.DOMAIN_SIZE_M,
               area_size_km=CFG.AREA_SIZE_KM
           )
           self.shadow_calculator.cloud_transmittance = CFG.CLOUD_TRANSMITTANCE
           self.shadow_calculator.shadow_fade_ms = CFG.SHADOW_FADE_MS
           def calculate_penumbra_width(altitude_m):
               base_width = max(60, altitude_m * math.tan(math.radians(0.27)))
               return min(base_width, 300)
           self.shadow_calculator.penumbra_width = calculate_penumbra_width(1000) / 1000
           self.shadow_calculator.spatial_cell_size = self.CELL_SIZE_KM
           print(f"Shadow calculator initialized (transmittance={self.shadow_calculator.cloud_transmittance}, fade={self.shadow_calculator.shadow_fade_ms}ms)")
           print(f"Panel spatial index built with {len(self.panel_cells)} cells")
           self._initialize_power_simulator()
       except Exception as e:
           print(f"Error during system initialization: {e}")
           import traceback
           traceback.print_exc()
           raise
   
   def _initialize_power_simulator(self):
       try:
           from power_simulator import PowerSimulator
           self.power_simulator = PowerSimulator(
               panel_df=self.panel_df,
               latitude=6.9271,
               longitude=79.8612
           )
           print("Power simulator initialized")
           if not hasattr(self.power_simulator, 'sunrise_hour'):
               self.power_simulator.sunrise_hour = 6.5
           if not hasattr(self.power_simulator, 'sunset_hour'):
               self.power_simulator.sunset_hour = 18.5
           if not hasattr(self.power_simulator, 'calculate_solar_position'):
               self.power_simulator.calculate_solar_position = lambda hour: {
                   "elevation": 90 - abs(12-hour)*7, 
                   "azimuth": (180 + (hour-12)*15) % 360
               }
           print(f"Solar generation hours: {self.power_simulator.sunrise_hour:.1f}h to {self.power_simulator.sunset_hour:.1f}h")
       except Exception as e:
           print(f"Error initializing power simulator: {e}")
           import traceback
           traceback.print_exc()
           from types import SimpleNamespace
           self.power_simulator = SimpleNamespace()
           self.power_simulator.calculate_power = lambda hour, coverage: {"total": 0.0}
           self.power_simulator.sunrise_hour = 6.5
           self.power_simulator.sunset_hour = 18.5
           self.power_simulator.calculate_solar_position = lambda hour: {"elevation": 90-abs(12-hour)*7, "azimuth": 180}
           self.power_simulator.calculate_clear_sky_power = lambda panel_id, hour: 5.0
   
   def step(self):
       self.frame_count += 1
       
       # Calculate time step
       current_time = time.time()
       dt = current_time - self.last_physics_time
       self.last_physics_time = current_time
       
       # Cap dt to avoid large jumps
       dt = min(dt, 0.1)
       
       self.accumulated_dt += dt
       while self.accumulated_dt >= CFG.PHYSICS_TIMESTEP:
           self.current_hour += self.timestep_minutes / 60.0 * (CFG.PHYSICS_TIMESTEP / (5.0 / 60.0))
           self.current_hour = self.current_hour % 24
           
           # Step the weather system with proper time step and current hour
           # Convert current_hour to seconds for simulation time
           sim_time_seconds = self.current_hour * 3600
           self.weather_system.step(t=self.frame_count, dt=CFG.PHYSICS_TIMESTEP, t_s=sim_time_seconds)
           
           self.accumulated_dt -= CFG.PHYSICS_TIMESTEP
           
       current_time = datetime.now()
       hour = int(self.current_hour)
       minute = int((self.current_hour % 1) * 60)
       timestamp = current_time.replace(hour=hour, minute=minute)
       solar_position = self.power_simulator.calculate_solar_position(self.current_hour)
       from cloud_simulation import collect_visible_ellipses
       cloud_ellipses = collect_visible_ellipses(self.weather_system.parcels)
       ellipses_for_shadow = [e[:7] for e in cloud_ellipses]
       panel_coverage = self.shadow_calculator.calculate_panel_coverage(
           ellipses_for_shadow, self.panel_df, solar_position, self.panel_cells
       )
       power_output = self.power_simulator.calculate_power(
           self.current_hour, panel_coverage
       )
       cloud_speed, cloud_direction, confidence = self.weather_system.get_avg_trajectory()
       if self.debug_mode and self.frame_count % 10 == 0:
           print(f"Hour: {self.current_hour:.2f}, Cloud count: {len(cloud_ellipses)}")
           print(f"Solar position: El={solar_position['elevation']:.1f}°, Az={solar_position['azimuth']:.1f}°")
           print(f"Cloud cover: {self.weather_system.current_cloud_cover_pct():.1f}%, Total power: {power_output.get('total', 0.0):.2f} kW")
           if cloud_speed is not None:
               print(f"Cloud movement: {cloud_speed:.1f} km/h, {cloud_direction:.0f}°")
       return {
           'time': timestamp,
           'cloud_ellipses': cloud_ellipses,
           'panel_coverage': panel_coverage,
           'power_output': power_output,
           'total_power': power_output.get('total', 0.0),
           'cloud_cover': self.weather_system.current_cloud_cover_pct(),
           'cloud_speed': cloud_speed,
           'cloud_direction': cloud_direction,
           'confidence': confidence,
           'solar_position': solar_position,
           'simulation_hour': self.current_hour  # Added simulation hour to result for reference
       }
   
   def render_clouds(self, screen, cloud_ellipses, cloud_positions, x_range, y_range, width, height):
       from pygame_rendereres.cloud_renderer import draw_cloud_trail, create_cloud_surface
       draw_cloud_trail(screen, cloud_positions, x_range, y_range, width, height)
       for ellipse_params in cloud_ellipses:
           cloud_surface, pos = create_cloud_surface(
               ellipse_params,
               CFG.DOMAIN_SIZE_M, CFG.AREA_SIZE_KM,
               width, height,
               x_range, y_range
           )
           screen.blit(cloud_surface, pos)
   
   def render_ui(self, screen, result, width, height, x_range, y_range):
       if self.font is None:
           self.font = pygame.font.SysFont('Arial', 16)
           self.large_font = pygame.font.SysFont('Arial', 20, bold=True)
           self.info_panel_bg = pygame.Surface((250, 150), pygame.SRCALPHA)
           self.info_panel_bg.fill((255, 255, 255, 180))
       cloud_ellipses = result.get('cloud_ellipses', [])
       panel_coverage = result.get('panel_coverage', {})
       power_output = result.get('power_output', {})
       total_power = power_output.get('total', 0.0)
       total_ac_power = power_output.get('total_ac', 0.0)
       baseline_total = power_output.get('baseline_total', 0.0)
       farm_reduction_pct = power_output.get('farm_reduction_pct', 0.0)
       cloud_cover = result.get('cloud_cover', 0.0)
       timestamp = result.get('time', datetime.now())
       cloud_speed = result.get('cloud_speed', None)
       cloud_direction = result.get('cloud_direction', None)
       confidence = result.get('confidence', 0)
       trajectory_source = result.get('trajectory_source', 'physics')
       solar_position = result.get('solar_position', {'elevation': 90, 'azimuth': 180})
       time_str = timestamp.strftime("%H:%M") if hasattr(timestamp, 'strftime') else f"{self.current_hour:.1f}h"
       info_dict = {
           "Time": time_str,
           "Sun": f"El={solar_position['elevation']:.1f}°, Az={solar_position['azimuth']:.1f}°",
           "Cloud Cover": f"{cloud_cover:.1f}%",
           "DC Power": f"{total_power:.1f} kW",
           "AC Output": f"{total_ac_power:.1f} kW",
           "Farm Reduction": f"{farm_reduction_pct:.1f}%"
       }
       self._draw_info_panel(screen, info_dict, "Simulation Info", (20, 20))
       if farm_reduction_pct > 30:
           self._draw_warning_bar(screen, farm_reduction_pct, (20, 170), 200, 30)
       self._draw_trajectory_info(screen, cloud_speed, cloud_direction, confidence, 
                               trajectory_source, (20, 210))
       affected_panels = [p for p, c in panel_coverage.items() if c > 0]
       self._draw_affected_panels_list(screen, affected_panels, len(self.panel_df), 
                                     power_output, (width - 350, 20))
       self._draw_time_slider(screen, self.current_hour, self.power_simulator.sunrise_hour,
                            self.power_simulator.sunset_hour, (width//2 - 200, height - 40), 400)
       help_text = "Controls: ESC-Quit, ↑↓-Wind Speed, ←→-Movement Speed"
       help_surface = self.font.render(help_text, True, (0, 0, 0), (255, 255, 255, 180))
       screen.blit(help_surface, (width//2 - help_surface.get_width()//2, height - 70))
   
   def _draw_info_panel(self, screen, info_dict, title, position):
       padding = 10
       line_height = 24
       num_lines = len(info_dict) + (1 if title else 0)
       panel_height = padding * 2 + line_height * num_lines
       panel_width = 250
       panel_rect = pygame.Rect(position[0], position[1], panel_width, panel_height)
       bg_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
       bg_surface.fill((255, 255, 255, 180))
       screen.blit(bg_surface, position)
       pygame.draw.rect(screen, (0, 0, 0), panel_rect, width=1)
       y_offset = position[1] + padding
       if title:
           title_surface = self.large_font.render(title, True, (0, 0, 0))
           screen.blit(title_surface, (position[0] + padding, y_offset))
           y_offset += line_height
       for key, value in info_dict.items():
           text = f"{key}: {value}"
           text_surface = self.font.render(text, True, (0, 0, 0))
           screen.blit(text_surface, (position[0] + padding, y_offset))
           y_offset += line_height
   
   def _draw_warning_bar(self, screen, reduction_pct, position, width, height):
       bar_rect = pygame.Rect(position[0], position[1], width, height)
       pygame.draw.rect(screen, (220, 220, 220), bar_rect)
       pygame.draw.rect(screen, (0, 0, 0), bar_rect, width=1)
       fill_width = int(min(100, reduction_pct) / 100 * width)
       if fill_width > 0:
           fill_rect = pygame.Rect(position[0], position[1], fill_width, height)
           if reduction_pct < 50:
               color = (255, 120, 0)
           else:
               color = (255, 0, 0)
           pygame.draw.rect(screen, color, fill_rect)
       text = f"Power Reduction Warning: {reduction_pct:.1f}%"
       text_surface = self.font.render(text, True, (0, 0, 0))
       text_rect = text_surface.get_rect(center=(position[0] + width//2, position[1] + height//2))
       screen.blit(text_surface, text_rect)
   
   def _draw_trajectory_info(self, screen, cloud_speed, cloud_direction, confidence, source, position):
       if cloud_speed is None or cloud_direction is None:
           text = "Cloud Movement: Not enough data"
           text_surface = self.font.render(text, True, (0, 0, 0))
           screen.blit(text_surface, position)
           return
       panel_width = 200
       panel_height = 170
       panel_rect = pygame.Rect(position[0], position[1], panel_width, panel_height)
       bg_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
       bg_surface.fill((255, 255, 255, 180))
       screen.blit(bg_surface, position)
       pygame.draw.rect(screen, (0, 0, 0), panel_rect, width=1)
       title_surface = self.large_font.render("Cloud Movement", True, (0, 0, 0))
       screen.blit(title_surface, (position[0] + 10, position[1] + 10))
       speed_text = f"Speed: {cloud_speed:.1f} km/h"
       speed_surface = self.font.render(speed_text, True, (0, 0, 0))
       screen.blit(speed_surface, (position[0] + 10, position[1] + 40))
       dir_text = f"Direction: {cloud_direction:.0f}°"
       dir_surface = self.font.render(dir_text, True, (0, 0, 0))
       screen.blit(dir_surface, (position[0] + 10, position[1] + 65))
       conf_text = f"Confidence: {int(confidence * 100)}%"
       conf_surface = self.font.render(conf_text, True, (0, 0, 0))
       screen.blit(conf_surface, (position[0] + 10, position[1] + 90))
       source_text = f"Source: {source.capitalize()}"
       source_color = (0, 120, 0) if source == 'power' else (0, 0, 120)
       source_surface = self.font.render(source_text, True, source_color)
       screen.blit(source_surface, (position[0] + 10, position[1] + 115))
       arrow_center = (
           position[0] + panel_width // 2,
           position[1] + panel_height - 30
       )
       arrow_length = 30
       direction_rad = math.radians(cloud_direction)
       end_x = arrow_center[0] + arrow_length * math.cos(direction_rad)
       end_y = arrow_center[1] - arrow_length * math.sin(direction_rad)
       pygame.draw.circle(screen, (230, 230, 230), arrow_center, arrow_length + 5)
       pygame.draw.circle(screen, (0, 0, 0), arrow_center, arrow_length + 5, width=1)
       compass_points = [
           ("N", 0, -1),
           ("E", 1, 0),
           ("S", 0, 1),
           ("W", -1, 0)
       ]
       small_font = pygame.font.SysFont('Arial', 10, bold=True)
       for label, dx, dy in compass_points:
           point_x = arrow_center[0] + (arrow_length + 15) * dx
           point_y = arrow_center[1] + (arrow_length + 15) * dy
           text = small_font.render(label, True, (0, 0, 0))
           text_rect = text.get_rect(center=(point_x, point_y))
           screen.blit(text, text_rect)
       if source == 'power':
           arrow_color = (0, 150, 0)
       else:
           if confidence > 0.7:
               arrow_color = (0, 0, 150)
           elif confidence > 0.3:
               arrow_color = (150, 150, 0)
           else:
               arrow_color = (150, 0, 0)
       pygame.draw.line(screen, arrow_color, arrow_center, (end_x, end_y), width=3)
       head_length = 10
       head_width = 6
       perp_x = math.sin(direction_rad)
       perp_y = math.cos(direction_rad)
       head_point1 = (
           end_x - head_length * math.cos(direction_rad) + head_width * perp_x,
           end_y + head_length * math.sin(direction_rad) + head_width * perp_y
       )
       head_point2 = (
           end_x - head_length * math.cos(direction_rad) - head_width * perp_x,
           end_y + head_length * math.sin(direction_rad) - head_width * perp_y
       )
       pygame.draw.polygon(screen, arrow_color, [(end_x, end_y), head_point1, head_point2])
   
   def _draw_affected_panels_list(self, screen, affected_panels, total_panels, power_output, position):
       if not affected_panels:
           return
       panel_width = 300
       panel_height = 250
       padding = 10
       line_height = 20
       panel_rect = pygame.Rect(position[0], position[1], panel_width, panel_height)
       bg_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
       bg_surface.fill((255, 255, 255, 180))
       screen.blit(bg_surface, position)
       pygame.draw.rect(screen, (0, 0, 0), panel_rect, width=1)
       affected_count = len(affected_panels)
       affected_pct = affected_count / total_panels * 100 if total_panels > 0 else 0
       title = f"Affected Panels: {affected_count}/{total_panels} ({affected_pct:.1f}%)"
       title_surface = self.large_font.render(title, True, (0, 0, 0))
       screen.blit(title_surface, (position[0] + padding, position[1] + padding))
       header = "Panel ID      Reduction      Power"
       header_surface = self.font.render(header, True, (0, 0, 0))
       header_rect = header_surface.get_rect()
       screen.blit(header_surface, (position[0] + padding, position[1] + padding + 30))
       line_y = position[1] + padding + 30 + header_rect.height + 5
       pygame.draw.line(
           screen, (0, 0, 0), 
           (position[0] + 5, line_y), 
           (position[0] + panel_width - 5, line_y),
           width=1
       )
       y_pos = line_y + 10
       for i, panel_id in enumerate(affected_panels[:10]):
           if panel_id not in power_output:
               continue
           power_data = power_output[panel_id]
           baseline = power_data.get('baseline', 0)
           current = power_data.get('final_power', 0)
           if baseline > 0:
               reduction = (baseline - current) / baseline * 100
           else:
               reduction = 0
           text = f"{panel_id:<10}   {reduction:>6.1f}%     {current:.2f} kW"
           if reduction > 50:
               color = (180, 0, 0)
           elif reduction > 20:
               color = (180, 120, 0)
           else:
               color = (0, 120, 0)
           text_surface = self.font.render(text, True, color)
           screen.blit(text_surface, (position[0] + padding, y_pos))
           y_pos += line_height
       if len(affected_panels) > 10:
           more_text = f"...and {len(affected_panels) - 10} more panels"
           more_surface = self.font.render(more_text, True, (100, 100, 100))
           screen.blit(more_surface, (position[0] + padding, y_pos + 5))
   
   def _draw_time_slider(self, screen, current_hour, sunrise_hour, sunset_hour, position, width):
       height = 20
       padding = 5
       day_start = 0
       day_end = 24
       day_length = day_end - day_start
       x_pos = position[0]
       y_pos = position[1]
       bg_rect = pygame.Rect(x_pos, y_pos, width, height)
       pygame.draw.rect(screen, (220, 220, 220), bg_rect, border_radius=height//2)
       pygame.draw.rect(screen, (0, 0, 0), bg_rect, width=1, border_radius=height//2)
       if sunrise_hour < sunset_hour:
           daylight_start = (sunrise_hour - day_start) / day_length * width
           daylight_width = (sunset_hour - sunrise_hour) / day_length * width
           daylight_rect = pygame.Rect(
               x_pos + daylight_start, 
               y_pos, 
               daylight_width, 
               height
           )
           gradient_surface = pygame.Surface((int(daylight_width), height), pygame.SRCALPHA)
           for x in range(int(daylight_width)):
               pos = x / daylight_width
               if pos < 0.5:
                   r = int(255 * (pos * 2))
                   g = int(200 * (pos * 2))
                   b = int(255 * (1 - pos))
               else:
                   adjusted_pos = (pos - 0.5) * 2
                   r = int(255 * (1 - adjusted_pos))
                   g = int(200 * (1 - adjusted_pos))
                   b = int(255 * adjusted_pos)
               pygame.draw.line(gradient_surface, (r, g, b), (x, 0), (x, height))
           screen.blit(gradient_surface, (x_pos + daylight_start, y_pos))
           pygame.draw.rect(screen, (0, 0, 0), bg_rect, width=1, border_radius=height//2)
       for hour in range(day_start, day_end + 1, 3):
           marker_x = x_pos + (hour - day_start) / day_length * width
           marker_height = height + 5
           pygame.draw.line(
               screen, (0, 0, 0),
               (marker_x, y_pos + height),
               (marker_x, y_pos + height + 5),
               width=1
           )
           hour_text = f"{hour:02d}:00"
           hour_surface = self.font.render(hour_text, True, (0, 0, 0))
           hour_rect = hour_surface.get_rect(center=(marker_x, y_pos + height + 15))
           screen.blit(hour_surface, hour_rect)
       if day_start <= current_hour <= day_end:
           current_x = x_pos + (current_hour - day_start) / day_length * width
           pointer_height = 15
           pointer_width = 10
           pointer_points = [
               (current_x, y_pos - 5),
               (current_x - pointer_width//2, y_pos - 5 - pointer_height),
               (current_x + pointer_width//2, y_pos - 5 - pointer_height)
           ]
           pygame.draw.polygon(screen, (200, 0, 0), pointer_points)
           hour = int(current_hour)
           minute = int((current_hour % 1) * 60)
           time_text = f"{hour:02d}:{minute:02d}"
           time_surface = self.font.render(time_text, True, (0, 0, 0))
           time_rect = time_surface.get_rect(center=(current_x, y_pos - 25))
           screen.blit(time_surface, time_rect)