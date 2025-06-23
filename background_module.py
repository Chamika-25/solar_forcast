"""
Background module for solar farm simulation
Handles panel layout, background grid, and static elements
"""
import os
import numpy as np
import pandas as pd
import pygame

def create_panel_dataframe(coordinates_path=None, csv_file=None, num_panels=None):
    """
    Create a DataFrame with solar panel information.
    
    Can load from:
    - coordinates_path/csv_file: Path to CSV with panel coordinates
    - num_panels: Number of panels to generate in a grid pattern
    
    Returns DataFrame with:
    - panel_id: Unique identifier
    - x_km, y_km: Position in kilometers
    - power_capacity: Maximum power in kW
    """
    if coordinates_path is not None or csv_file is not None:
        # Load from file
        path = coordinates_path if coordinates_path is not None else csv_file
        try:
            df = pd.read_csv(path)
            # Ensure required columns exist
            if 'panel_id' not in df.columns:
                df['panel_id'] = [f"P{i+1:03d}" for i in range(len(df))]
            if 'power_capacity' not in df.columns:
                df['power_capacity'] = 5.0  # Default capacity in kW
                
            print(f"Loaded {len(df)} panels from {path}")
            return df
        except Exception as e:
            print(f"Error loading panel coordinates from {path}: {e}")
            print("Falling back to generated panels")
    
    # Generate panels in a grid pattern
    if num_panels is None:
        num_panels = 36  # Default to 6x6 grid
    
    # Determine grid dimensions (approximate square)
    grid_size = int(np.ceil(np.sqrt(num_panels)))
    
    # Generate coordinates
    spacing = 3.0  # km between panels
    margin = 5.0   # km from edges
    
    panel_data = []
    panel_count = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            if panel_count < num_panels:
                x = margin + i * spacing
                y = margin + j * spacing
                
                # Add some randomization
                x += np.random.uniform(-0.5, 0.5)
                y += np.random.uniform(-0.5, 0.5)
                
                panel_data.append({
                    'panel_id': f"P{panel_count+1:03d}",
                    'x_km': x,
                    'y_km': y,
                    'power_capacity': np.random.uniform(4.0, 6.0)  # 4-6 kW capacity
                })
                panel_count += 1
    
    df = pd.DataFrame(panel_data)
    print(f"Generated {len(df)} panels in a grid pattern")
    return df

def setup_background(screen, width, height, x_range, y_range, grid_color=(204, 204, 204)):
    """
    Draw the background grid and coordinate system for the simulation.
    
    Args:
        screen: Pygame surface to draw on
        width, height: Screen dimensions
        x_range, y_range: Coordinate ranges in km
        grid_color: Color for grid lines
    """
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
    
    # Axis titles
    x_label = font.render("Distance (km)", True, (0, 0, 0))
    screen.blit(x_label, (width // 2 - 40, height - 20))
    
    y_label = font.render("Distance (km)", True, (0, 0, 0))
    y_label = pygame.transform.rotate(y_label, 90)
    screen.blit(y_label, (5, height // 2 - 40))

def km_to_screen_coords(x_km, y_km, x_range, y_range, screen_width, screen_height):
    """Convert km coordinates to screen pixels"""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    screen_x = int((x_km - x_range[0]) / range_x * screen_width)
    screen_y = int((y_km - y_range[0]) / range_y * screen_height)
    return screen_x, screen_y

def screen_to_km_coords(screen_x, screen_y, x_range, y_range, screen_width, screen_height):
    """Convert screen pixels to km coordinates"""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    x_km = x_range[0] + (screen_x / screen_width) * range_x
    y_km = y_range[0] + (screen_y / screen_height) * range_y
    return x_km, y_km

def draw_solar_panels(screen, panel_df, panel_coverage, power_output, x_range, y_range, width, height):
    """
    Draw solar panels with power output visualization.
    
    Args:
        screen: Pygame surface to draw on
        panel_df: DataFrame with panel information
        panel_coverage: Dict mapping panel_id to coverage percentage
        power_output: Dict mapping panel_id to power output info
        x_range, y_range: Coordinate ranges in km
        width, height: Screen dimensions
    
    Returns:
        List of affected panels (those with coverage > 0)
    """
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    panel_size_km = 0.4  # Default panel size
    panel_size_px = int(panel_size_km / range_x * width)
    
    affected_panels = []
    
    for _, row in panel_df.iterrows():
        panel_id = row["panel_id"]
        x_km = row["x_km"]
        y_km = row["y_km"]
        
        # Convert to screen coordinates
        x_px = int((x_km - x_range[0]) / range_x * width)
        y_px = int((y_km - y_range[0]) / range_y * height)
        
        # Get panel coverage
        coverage = panel_coverage.get(panel_id, 0.0)
        if coverage > 0:
            affected_panels.append(panel_id)
        
        # Get power output
        power_data = power_output.get(panel_id, {})
        power_value = power_data.get('final_power', 0)
        max_power = power_data.get('baseline', 1.0)
        
        # Determine panel color based on power percentage
        if max_power > 0:
            power_pct = power_value / max_power
        else:
            power_pct = 0
        
        if power_pct > 0.8:
            panel_color = (0, 0, 200)  # Blue - high power
        elif power_pct > 0.5:
            panel_color = (0, 150, 0)  # Green - medium power
        elif power_pct > 0.2:
            panel_color = (200, 150, 0)  # Orange - low power
        else:
            panel_color = (200, 0, 0)  # Red - very low power
        
        # Draw panel
        pygame.draw.rect(
            screen,
            panel_color,
            (x_px - panel_size_px//2, y_px - panel_size_px//2, panel_size_px, panel_size_px)
        )
        
        # Add shadow overlay if covered by clouds
        if coverage > 0.05:
            shadow_size = int(panel_size_px * 0.8)
            shadow_color = (0, 0, 0, int(coverage * 180))  # Alpha based on coverage
            
            # Create shadow surface with alpha
            shadow_surface = pygame.Surface((shadow_size, shadow_size), pygame.SRCALPHA)
            shadow_surface.fill(shadow_color)
            
            # Blit shadow onto panel
            screen.blit(shadow_surface, (x_px - shadow_size//2, y_px - shadow_size//2))
        
        # Draw panel ID if not too many panels
        if len(panel_df) < 60:
            font = pygame.font.SysFont('Arial', 9 if len(panel_df) > 30 else 12)
            label = font.render(panel_id, True, (255, 255, 255))
            label_rect = label.get_rect(center=(x_px, y_px))
            screen.blit(label, label_rect)
    
    return affected_panels

def create_ui_element(text, position, size, font_size=16, bg_color=(255, 255, 255, 200), text_color=(0, 0, 0)):
    """Create a UI element with text"""
    surface = pygame.Surface(size, pygame.SRCALPHA)
    surface.fill(bg_color)
    
    font = pygame.font.SysFont('Arial', font_size)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(size[0]//2, size[1]//2))
    
    surface.blit(text_surface, text_rect)
    
    return surface