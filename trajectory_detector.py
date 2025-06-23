"""
Trajectory Detector for Solar Farm Simulation
Infers cloud movement from power output patterns
"""
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import math

class PowerSnapshot:
    """Stores power data for a specific timestamp"""
    def __init__(self, timestamp, panel_power_dict):
        self.timestamp = timestamp
        self.panel_power = panel_power_dict.copy()
        
        # Calculate affected panels (those with reduced power)
        self.affected_panels = set()
        self.affected_positions = []
        
        for panel_id, power in panel_power_dict.items():
            if panel_id.startswith('P') and power < 5.0:  # Assuming 5.0 kW is full capacity
                self.affected_panels.add(panel_id)
    
    def set_panel_positions(self, panel_df):
        """Set panel positions for affected panels"""
        self.affected_positions = []
        
        for panel_id in self.affected_panels:
            panel_row = panel_df[panel_df['panel_id'] == panel_id]
            if not panel_row.empty:
                x = panel_row.iloc[0]['x_km']
                y = panel_row.iloc[0]['y_km']
                self.affected_positions.append((x, y))
    
    def get_centroid(self):
        """Calculate centroid of affected panels"""
        if not self.affected_positions:
            return None
        
        x_sum = sum(p[0] for p in self.affected_positions)
        y_sum = sum(p[1] for p in self.affected_positions)
        
        return (x_sum / len(self.affected_positions), 
                y_sum / len(self.affected_positions))
    
    def get_affected_count(self):
        """Get number of affected panels"""
        return len(self.affected_panels)

class TrajectoryDetector:
    """
    Detect cloud trajectory from power output patterns.
    """
    
    def __init__(self, panel_df, buffer_seconds=60):
        """
        Initialize trajectory detector.
        
        Args:
            panel_df: DataFrame with panel information
            buffer_seconds: Number of seconds to buffer power data
        """
        self.panel_df = panel_df
        self.buffer_seconds = buffer_seconds
        
        # Initialize power history buffer
        self.power_history = deque(maxlen=int(buffer_seconds * 1))  # 1 sample per second
        self.last_update_time = None
        
        # Track power deltas
        self.power_deltas = {}
        
        # Current trajectory info
        self.current_trajectory = None
    
    def update_from_power(self, panel_power_dict, timestamp):
        """
        Update trajectory inference from power data.
        
        Args:
            panel_power_dict: Dictionary mapping panel_id to power (kW)
            timestamp: Current timestamp
        
        Returns:
            Dict with inferred trajectory info (speed, direction, confidence)
        """
        # Skip if no power data
        if not panel_power_dict:
            return None
        
        # Only update once per second to avoid too many snapshots
        if self.last_update_time and (timestamp - self.last_update_time).total_seconds() < 1.0:
            return self.current_trajectory
        
        # Create power snapshot
        snapshot = PowerSnapshot(timestamp, panel_power_dict)
        snapshot.set_panel_positions(self.panel_df)
        
        # Only add snapshot if there are affected panels
        if snapshot.get_affected_count() > 0:
            self.power_history.append(snapshot)
            self.last_update_time = timestamp
        
        # Need at least 2 snapshots for trajectory
        if len(self.power_history) < 2:
            return None
        
        # Calculate trajectory
        return self._calculate_trajectory()
    
    def _calculate_trajectory(self):
        """
        Calculate cloud trajectory from power history.
        
        Returns:
            Dict with trajectory info (speed, direction, confidence)
        """
        # Get earliest and latest snapshots with affected panels
        earliest = None
        latest = None
        
        for snapshot in self.power_history:
            if snapshot.get_affected_count() > 0:
                if earliest is None or snapshot.timestamp < earliest.timestamp:
                    earliest = snapshot
                if latest is None or snapshot.timestamp > latest.timestamp:
                    latest = snapshot
        
        # Ensure we have valid snapshots
        if earliest is None or latest is None or earliest == latest:
            return None
        
        # Get centroids
        earliest_centroid = earliest.get_centroid()
        latest_centroid = latest.get_centroid()
        
        if earliest_centroid is None or latest_centroid is None:
            return None
        
        # Calculate direction vector
        dx = latest_centroid[0] - earliest_centroid[0]
        dy = latest_centroid[1] - earliest_centroid[1]
        
        # Skip if movement is too small
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return None
        
        # Calculate direction angle (0 = East, 90 = North)
        direction = math.degrees(math.atan2(dy, dx)) % 360
        
        # Calculate speed
        time_diff = (latest.timestamp - earliest.timestamp).total_seconds() / 3600  # hours
        distance = math.sqrt(dx*dx + dy*dy)  # km
        
        if time_diff <= 0:
            return None
            
        speed = distance / time_diff  # km/h
        
        # Calculate confidence based on number of affected panels
        affected_count = max(earliest.get_affected_count(), latest.get_affected_count())
        confidence = min(1.0, affected_count / 5)  # 5+ panels = full confidence
        
        # Store trajectory info
        self.current_trajectory = {
            'speed': speed,
            'direction': direction,
            'confidence': confidence,
            'affected_count': affected_count,
            'time_span': time_diff * 60  # minutes
        }
        
        return self.current_trajectory
    
    def get_current_trajectory(self):
        """Get current trajectory info"""
        return self.current_trajectory