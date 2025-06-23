import numpy as np
from datetime import datetime, timedelta
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CloudThresholdTracker:
    """
    Simple cloud tracker using power generation thresholds.
    This is conceptually similar to an array of pyranometers.
    """
    
    def __init__(self, panel_df, threshold_pct=20):
        """
        Initialize the tracker.
        
        Args:
            panel_df: DataFrame with panel positions
            threshold_pct: Power reduction percentage to consider as "cloud detected"
        """
        self.panel_positions = {
            row['panel_id']: (row['x_km'], row['y_km']) 
            for _, row in panel_df.iterrows()
        }
        self.threshold_pct = threshold_pct
        
        # Track cloud crossing events
        self.cloud_events = deque(maxlen=100)
        self.panel_states = {}  # Current state of each panel (cloudy or clear)
        self.last_event_time = None
        self.panel_df = panel_df
        
        # Status display
        self.status_messages = deque(maxlen=10)
        
    def update(self, timestamp, power_output):
        """
        Update cloud tracking with new power data.
        
        Args:
            timestamp: Current timestamp
            power_output: Dictionary of power output details per panel
        """
        state_changes = []
        
        for panel_id, data in power_output.items():
            baseline = data.get('baseline', 0)
            actual = data.get('final_power', 0)
            
            if baseline > 0:
                # Calculate power reduction percentage
                reduction_pct = (baseline - actual) / baseline * 100
                
                # Determine current state
                is_cloudy = reduction_pct >= self.threshold_pct
                
                # Check if state changed
                if panel_id not in self.panel_states:
                    self.panel_states[panel_id] = is_cloudy
                elif self.panel_states[panel_id] != is_cloudy:
                    # State changed! Record this event
                    event = {
                        'panel_id': panel_id,
                        'timestamp': timestamp,
                        'event_type': 'cloud_arrival' if is_cloudy else 'cloud_departure',
                        'position': self.panel_positions.get(panel_id, (0, 0)),
                        'reduction_pct': reduction_pct
                    }
                    self.cloud_events.append(event)
                    self.panel_states[panel_id] = is_cloudy
                    state_changes.append(event)
        
        # Calculate trajectory if we have enough events
        speed, direction, confidence = self._calculate_trajectory()
        
        # Log status message if there were state changes
        if state_changes:
            if any(e['event_type'] == 'cloud_arrival' for e in state_changes):
                arrivals = [e['panel_id'] for e in state_changes if e['event_type'] == 'cloud_arrival']
                if speed is not None:
                    msg = f"Cloud detected on panel(s) {', '.join(arrivals)}. Moving at {speed:.1f}km/h {direction:.0f}°"
                else:
                    msg = f"Cloud detected on panel(s) {', '.join(arrivals)}. Calculating trajectory..."
                self.status_messages.append(msg)
            
            if any(e['event_type'] == 'cloud_departure' for e in state_changes):
                departures = [e['panel_id'] for e in state_changes if e['event_type'] == 'cloud_departure']
                self.status_messages.append(f"Cloud departed from panel(s) {', '.join(departures)}")
                
        return speed, direction, confidence
                
    def _calculate_trajectory(self):
        """Calculate cloud trajectory from recent events."""
        # Need at least 2 events to calculate trajectory
        if len(self.cloud_events) < 2:
            return None, None, 0  # speed, direction, confidence
        
        # Look at most recent arrival events
        arrivals = [e for e in self.cloud_events 
                   if e['event_type'] == 'cloud_arrival']
        
        if len(arrivals) < 2:
            return None, None, 0
            
        # Sort by timestamp
        arrivals.sort(key=lambda e: e['timestamp'])
        
        # Calculate between most recent arrivals
        recent = arrivals[-2:]
        
        # Extract positions and times
        p1 = recent[0]['position']
        p2 = recent[1]['position']
        t1 = recent[0]['timestamp']
        t2 = recent[1]['timestamp']
        
        # Calculate time difference in hours
        time_diff_seconds = (t2 - t1).total_seconds()
        time_diff_hours = time_diff_seconds / 3600
        
        if time_diff_hours <= 0:
            return None, None, 0
        
        # Calculate distance in km
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Calculate speed
        speed = distance / time_diff_hours  # km/h
        
        # Calculate direction (in degrees, 0 = East, 90 = North)
        direction = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 360
        
        # Simple confidence value (higher for more recent events)
        confidence = min(1.0, len(arrivals) / 5)
        
        return speed, direction, confidence
    
    def get_cloud_status(self):
        """Get dictionary of which panels are currently under cloud."""
        return self.panel_states.copy()
    
    def get_recent_events(self, count=5):
        """Get recent cloud events for display."""
        return list(self.cloud_events)[-count:]
    
    def visualize_cloud_status(self, ax, panel_size_km=0.4):
        """
        Visualize current cloud status on the provided matplotlib axes.
        
        Args:
            ax: Matplotlib axes to draw on
            panel_size_km: Size of solar panels in km
        """
        # Clear existing elements
        for artist in ax.get_children():
            if isinstance(artist, patches.Rectangle) and getattr(artist, 'is_threshold_marker', False):
                artist.remove()
        
        # Draw cloud status
        status = self.get_cloud_status()
        
        for _, panel in self.panel_df.iterrows():
            panel_id = panel["panel_id"]
            x_km = panel["x_km"]
            y_km = panel["y_km"]
            
            # Default to clear if not in status dict
            is_cloudy = status.get(panel_id, False)
            
            # Draw a border that shows threshold status
            if is_cloudy:
                # Thicker red border for cloudy panels
                rect = patches.Rectangle(
                    (x_km - panel_size_km/2 - 0.05, y_km - panel_size_km/2 - 0.05),
                    panel_size_km + 0.1, panel_size_km + 0.1,
                    linewidth=2, edgecolor='red', facecolor='none',
                    alpha=0.8, zorder=20
                )
                rect.is_threshold_marker = True  # Custom attribute to identify our markers
                ax.add_patch(rect)
        
        # Show trajectory if available
        speed, direction, confidence = self._calculate_trajectory()
        if speed is not None and confidence > 0.2:
            # Find center point of all panels
            positions = np.array(list(self.panel_positions.values()))
            center_x, center_y = np.mean(positions, axis=0)
            
            # Draw arrow showing trajectory
            arrow_length = min(2.0, speed * 0.05)
            dx = arrow_length * np.cos(np.radians(direction))
            dy = arrow_length * np.sin(np.radians(direction))
            
            arrow = ax.arrow(center_x, center_y, dx, dy, 
                            head_width=0.3, head_length=0.5, 
                            fc='black', ec='black', 
                            alpha=min(0.8, confidence + 0.3),
                            zorder=25, linewidth=3)
            
        # Return the most recent status message
        if self.status_messages:
            return self.status_messages[-1]
        return "No cloud events detected yet"