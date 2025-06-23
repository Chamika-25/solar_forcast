"""
Panel Layout Module for Solar Farm Simulation
Loads and provides access to static panel configuration
"""
import csv
import os
from collections import defaultdict

# Load panels from CSV on import
PANELS = {}

# Try to load the panel data from CSV
csv_path = "extended_coordinates.csv"
if os.path.exists(csv_path):
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            panel_id = row['panel_id']
            PANELS[panel_id] = {
                "x_km": float(row['x_km']),
                "y_km": float(row['y_km']),
                "power_capacity": 5.0  # Default capacity
            }
    print(f"Loaded {len(PANELS)} panels from {csv_path}")
else:
    # Fallback: create a minimal panel set
    for i in range(1, 37):
        panel_id = f"P{i:03d}"
        x = (i-1) % 6 * 5 + 5  # 5, 10, 15, 20, 25, 30
        y = (i-1) // 6 * 5 + 5  # 5, 10, 15, 20, 25, 30
        PANELS[panel_id] = {
            "x_km": float(x),
            "y_km": float(y),
            "power_capacity": 5.0
        }
    print(f"Warning: {csv_path} not found. Created {len(PANELS)} default panels")

def build_panel_cells(cell_size_km=2.0):
    """
    Build a spatial index for panels.
    
    Args:
        cell_size_km: Size of each spatial cell in km
        
    Returns:
        defaultdict mapping (grid_x, grid_y) to lists of panel_ids
    """
    panel_cells = defaultdict(list)
    
    for panel_id, data in PANELS.items():
        x_km = data["x_km"]
        y_km = data["y_km"]
        
        # Calculate grid cell coordinates
        grid_x = int(x_km / cell_size_km)
        grid_y = int(y_km / cell_size_km)
        
        # Add panel to appropriate cell
        panel_cells[(grid_x, grid_y)].append(panel_id)
    
    return panel_cells

# Create a DataFrame-like accessor for compatibility with existing code
class PanelDataFrame:
    """DataFrame-like access to panel data for backward compatibility"""
    
    def __init__(self):
        self._panels = PANELS
    
    def __len__(self):
        return len(self._panels)
    
    def iterrows(self):
        """Simulate pandas DataFrame.iterrows()"""
        for panel_id, data in self._panels.items():
            yield None, {
                "panel_id": panel_id,
                "x_km": data["x_km"],
                "y_km": data["y_km"],
                "power_capacity": data.get("power_capacity", 5.0)
            }

# Create a singleton instance for import
panel_df = PanelDataFrame()