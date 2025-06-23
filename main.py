# main.py
#!/usr/bin/env python3
"""
Solar Farm Cloud Simulation - Main entry point
Pygame visualization with enhanced cloud physics
"""
import argparse
import time
import traceback
import pygame
import math
import numpy as np
import os
import sys
import sim_config as CFG

# Add the renderer path to the system path
sys.path.append('C:/Users/Suchira_Garusinghe/Desktop/Simulation/simulation8.5.0/pygame_rendereres')

def safely_get_cloud_positions(cloud_ellipses, old_positions):
    """
    Safely extract cloud positions from ellipses with error handling.
    
    Args:
        cloud_ellipses: List of cloud ellipse parameters
        old_positions: Previous cloud positions for continuity
    
    Returns:
        List of (x_km, y_km) positions
    """
    positions = old_positions.copy() if old_positions else []
    
    try:
        # Only process if we have ellipses
        if cloud_ellipses and len(cloud_ellipses) > 0:
            # Get center of the first cloud
            if len(cloud_ellipses[0]) >= 2:
                x, y = cloud_ellipses[0][0], cloud_ellipses[0][1]
                km_x = x / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                km_y = y / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                
                # Add new position
                positions.append((km_x, km_y))
                
                # Limit trail length
                max_trail_length = 20  # Increased for better visualization
                while len(positions) > max_trail_length:
                    positions.pop(0)
    except Exception as e:
        print(f"Error processing cloud positions: {e}")
    
    return positions

def setup_background(screen, width, height, x_range, y_range, grid_color=(204, 204, 204)):
    """Draw the background grid and coordinate system for the simulation."""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    
    # Determine grid spacing based on view range
    grid_interval_km = 5.0 if range_x > 20 else 2.0
    
    # Draw grid lines
    for x_km in np.arange(x_range[0], x_range[1] + grid_interval_km, grid_interval_km):
        x_px = int((x_km - x_range[0]) / range_x * width)
        pygame.draw.line(screen, grid_color, (x_px, 0), (x_px, height), 1)
    
    for y_km in np.arange(y_range[0], y_range[1] + grid_interval_km, grid_interval_km):
        y_px = int((y_km - y_range[0]) / range_y * height)
        pygame.draw.line(screen, grid_color, (0, y_px), (width, y_px), 1)
    
    # Draw coordinate labels
    font = pygame.font.SysFont('Arial', 12)
    
    # Determine label spacing
    tick_interval = 10 if range_x > 20 else 5
    
    # X-axis labels
    for x_km in np.arange(int(x_range[0]), int(x_range[1])+1, tick_interval):
        x_px = int((x_km - x_range[0]) / range_x * width)
        label = font.render(f"{int(x_km)}", True, (0, 0, 0))
        screen.blit(label, (x_px - 5, height - 20))
    
    # Y-axis labels
    for y_km in np.arange(int(y_range[0]), int(y_range[1])+1, tick_interval):
        y_px = int((y_km - y_range[0]) / range_y * height)
        label = font.render(f"{int(y_km)}", True, (0, 0, 0))
        screen.blit(label, (5, y_px - 10))

def run_simulation():
    """Main entry point for the simulation"""
    parser = argparse.ArgumentParser(description='Solar Farm Cloud Trajectory Simulation')
    parser.add_argument('--start-time', type=float, default=11.0, help='Starting hour (default 11:00 AM)')
    parser.add_argument('--time-step', type=int, default=5, help='Time step in minutes')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging')
    parser.add_argument('--fps', type=int, default=CFG.FPS, help=f'Animation frames per second (default: {CFG.FPS})')
    parser.add_argument('--width', type=int, default=1200, help='Window width (pygame only)')
    parser.add_argument('--height', type=int, default=900, help='Window height (pygame only)')
    parser.add_argument('--single-cloud', action='store_true', help='Enable single cloud mode')
    parser.add_argument('--spawn-prob', type=float, help='Override cloud spawn probability')
    parser.add_argument('--no-logging', action='store_true', help='Disable power data logging to save disk space')
    parser.add_argument('--panel-file', type=str, default="extended_coordinates.csv", help='CSV file with panel coordinates')
    args = parser.parse_args()
    
    print("===== SOLAR FARM CLOUD TRAJECTORY ANALYSIS =====")
    print(f"Starting simulation with solar generation patterns...")
    print(f"Solar generation hours: 6:30 AM to 6:30 PM")
    print(f"Time step: {args.time_step} minutes")
    print(f"Animation FPS: {args.fps}")
    print(f"Debug mode: {args.debug}")
    print("=" * 50)
    
    try:
        # Apply CLI overrides to config
        if args.single_cloud:
            CFG.SINGLE_CLOUD_MODE = True
            print("Single cloud mode enabled")
        if args.spawn_prob is not None:
            CFG.SPAWN_PROBABILITY = args.spawn_prob
            print(f"Cloud spawn probability set to {CFG.SPAWN_PROBABILITY}")
        
        # Import cloud simulation modules after config changes
        from cloud_simulation import WeatherSystem, collect_visible_ellipses
        from pygame_rendereres.cloud_renderer import create_cloud_surface
        from background_module import create_panel_dataframe, draw_solar_panels
        
        print(f"Settings configured for cloud movement:")
        print(f"Single Cloud Mode: {CFG.SINGLE_CLOUD_MODE}")
        print(f"MAX_PARCELS={CFG.MAX_PARCELS}, SPAWN_PROBABILITY={CFG.SPAWN_PROBABILITY}")
        print(f"BASE_WIND_SPEED={CFG.BASE_WIND_SPEED} m/s, WIND_UPDATE_SEC={CFG.WIND_UPDATE_SEC}s")
        
        # Load panel data from the specified CSV file
        panel_df = None
        try:
            # Check if the panel file exists
            if os.path.exists(args.panel_file):
                print(f"Loading panels from {args.panel_file}")
                panel_df = create_panel_dataframe(coordinates_path=args.panel_file)
                print(f"Loaded {len(panel_df)} panels from {args.panel_file}")
            else:
                # Try looking in the current directory for extended_coordinates.csv
                if os.path.exists("extended_coordinates.csv"):
                    print("Loading panels from extended_coordinates.csv")
                    panel_df = create_panel_dataframe(coordinates_path="extended_coordinates.csv")
                    print(f"Loaded {len(panel_df)} panels")
                else:
                    print("Panel coordinates file not found. Using default coordinates.")
                    panel_df = create_panel_dataframe(num_panels=36)
                    print(f"Created panel dataframe with {len(panel_df)} panels")
        except Exception as e:
            print(f"Error loading panel data: {e}")
            print("Generating default panel data")
            
            # Create a simple default panel dataframe
            import pandas as pd
            panel_data = []
            
            for i in range(6):
                for j in range(6):
                    panel_data.append({
                        'panel_id': f"P{i*6+j+1:03d}",
                        'x_km': 10 + i * 5,
                        'y_km': 10 + j * 5,
                        'power_capacity': 5.0
                    })
            
            panel_df = pd.DataFrame(panel_data)
            print(f"Created default panel dataframe with {len(panel_df)} panels")
        
        # Calculate the bounds of the panel layout for viewport setting
        try:
            x_values = panel_df['x_km'].values
            y_values = panel_df['y_km'].values
            
            # Expand the visible area to include all panels plus margin
            panel_min_x = max(0, np.min(x_values) - 5)
            panel_max_x = min(CFG.AREA_SIZE_KM, np.max(x_values) + 5)
            panel_min_y = max(0, np.min(y_values) - 5)
            panel_max_y = min(CFG.AREA_SIZE_KM, np.max(y_values) + 5)
            
            print(f"Panel layout bounds: X: {panel_min_x:.1f} to {panel_max_x:.1f}, Y: {panel_min_y:.1f} to {panel_max_y:.1f}")
        except Exception as e:
            print(f"Error calculating panel bounds: {e}")
            panel_min_x, panel_max_x = 0, CFG.AREA_SIZE_KM
            panel_min_y, panel_max_y = 0, CFG.AREA_SIZE_KM
        
        # Initialize simulation controller
        from simulation_controller import SimulationController
        controller = SimulationController(
            start_time=args.start_time,
            timestep_minutes=args.time_step,
            debug_mode=args.debug,
            panel_df=panel_df
        )
        
        # Set up view area to cover the whole domain
        x_range = (0, CFG.AREA_SIZE_KM)
        y_range = (0, CFG.AREA_SIZE_KM)
        
        # Run simulation with Pygame
        run_with_pygame(controller, x_range, y_range, args.width, args.height, args.fps, args.debug)
        
        print("Simulation completed successfully.")
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    except Exception as e:
        print(f"\nSimulation error: {e}")
        traceback.print_exc()

def run_with_pygame(controller, x_range, y_range, width, height, fps, debug_mode):
    """Run simulation with Pygame backend"""
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("Solar Farm Cloud Simulation")
    clock = pygame.time.Clock()
    
    # Colors
    SKY_COLOR = (230, 242, 255)
    
    # Performance metrics
    frame_times = []
    last_fps_update = time.time()
    
    # Main simulation loop
    running = True
    frame_count = 0
    
    # Create font for display
    font = pygame.font.SysFont('Arial', 16)
    
    # Cloud position tracking - INITIALIZE EMPTY LIST
    cloud_positions = []
    
    try:
        while running:
            start_time = time.time()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        # Increase wind speed
                        CFG.BASE_WIND_SPEED = min(10.0, CFG.BASE_WIND_SPEED + 0.5)
                        print(f"Wind speed increased to {CFG.BASE_WIND_SPEED:.1f} m/s")
                    elif event.key == pygame.K_DOWN:
                        # Decrease wind speed
                        CFG.BASE_WIND_SPEED = max(0.5, CFG.BASE_WIND_SPEED - 0.5)
                        print(f"Wind speed decreased to {CFG.BASE_WIND_SPEED:.1f} m/s")
                    elif event.key == pygame.K_LEFT:
                        # Decrease movement multiplier
                        CFG.MOVEMENT_MULTIPLIER = max(1.0, CFG.MOVEMENT_MULTIPLIER - 0.5)
                        print(f"Movement multiplier decreased to {CFG.MOVEMENT_MULTIPLIER:.1f}")
                    elif event.key == pygame.K_RIGHT:
                        # Increase movement multiplier
                        CFG.MOVEMENT_MULTIPLIER = min(15.0, CFG.MOVEMENT_MULTIPLIER + 0.5)
                        print(f"Movement multiplier increased to {CFG.MOVEMENT_MULTIPLIER:.1f}")
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    width, height = event.size
                    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            
            # Simulation step
            try:
                result = controller.step()
                # Extract results
                cloud_ellipses = result.get('cloud_ellipses', [])
                panel_coverage = result.get('panel_coverage', {})
                power_output = result.get('power_output', {})
                total_power = power_output.get('total', 0.0)
                cloud_cover = result.get('cloud_cover', 0.0)
                timestamp = result.get('time', None)
                cloud_speed = result.get('cloud_speed', None)
                cloud_direction = result.get('cloud_direction', None)
                confidence = result.get('confidence', 0)
                
                # Track ellipse positions for trail (UPDATED TO USE SAFER FUNCTION)
                try:
                    cloud_positions = safely_get_cloud_positions(cloud_ellipses, cloud_positions)
                except Exception as e:
                    print(f"Error tracking cloud positions: {e}")
                    # Keep existing positions if there's an error
                    pass
                
            except Exception as e:
                print(f"Error in simulation step: {e}")
                if debug_mode:
                    traceback.print_exc()
                # Default empty values for error case
                result = {
                    'cloud_ellipses': [],
                    'panel_coverage': {},
                    'power_output': {'total': 0.0},
                    'cloud_cover': 0.0
                }
                cloud_ellipses = []
                panel_coverage = {}
                power_output = {'total': 0.0}
            
            # Render frame
            screen.fill(SKY_COLOR)
            
            # Draw background grid
            setup_background(screen, width, height, x_range, y_range)
            
            # Draw solar panels
            from background_module import draw_solar_panels
            draw_solar_panels(
                screen, controller.panel_df, 
                panel_coverage, power_output, 
                x_range, y_range, width, height
            )
            
            # Draw clouds using controller's render_clouds method
            controller.render_clouds(
                screen, cloud_ellipses, cloud_positions,
                x_range, y_range, width, height
            )
            
            # Draw UI elements
            controller.render_ui(screen, result, width, height, x_range, y_range)
            
            # Draw FPS counter
            current_fps = clock.get_fps()
            fps_text = f"FPS: {current_fps:.1f}"
            try:
                fps_surface = font.render(fps_text, True, (0, 0, 0), (255, 255, 255, 180))
                screen.blit(fps_surface, (width - 100, 10))
            except:
                pass
            
            # Update display
            pygame.display.flip()
            
            # Cap frame rate
            clock.tick(fps)
            
            # Measure frame time
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 100:
                frame_times.pop(0)
            
            # Print debug info occasionally
            if debug_mode and frame_count % 60 == 0:
                avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
                print(f"Frame {frame_count}: Avg frame time: {avg_frame_time*1000:.1f}ms, FPS: {current_fps:.1f}")
                print(f"Cloud count: {len(cloud_ellipses)}, Cloud cover: {cloud_cover:.1f}%, Total power: {total_power:.2f} kW")
                if cloud_speed is not None:
                    print(f"Cloud movement: {cloud_speed:.1f} km/h, {cloud_direction:.0f}°, Confidence: {confidence:.2f}")
            
            frame_count += 1
    
    finally:
        # Clean up resources
        if hasattr(controller, 'power_simulator') and hasattr(controller.power_simulator, 'close'):
            controller.power_simulator.close()
        pygame.quit()

if __name__ == "__main__":
    run_simulation()