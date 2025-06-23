"""
Power Integration Module for Solar Farm Simulation
Handles power calculations, averaging, and logging
"""
import numpy as np
import os
import pandas as pd
from datetime import datetime
from collections import deque
import csv

class PowerIntegration:
    """
    Integrate power generation with cloud simulation.
    Handles power calculations, statistics, and logging.
    """
    
    def __init__(self, controller=None, settings=None):
        """
        Initialize power integration.
        
        Args:
            controller: SimulationController instance
            settings: Settings object with configuration parameters
        """
        self.controller = controller
        
        # Configure settings
        self.log_dir = "./logs"
        self.enable_logging = True
        
        if settings:
            if hasattr(settings, 'LOG_DIR'):
                self.log_dir = settings.LOG_DIR
            if hasattr(settings, 'ENABLE_POWER_LOGGING'):
                self.enable_logging = settings.ENABLE_POWER_LOGGING
            
            # Configure shadow calculator
            if hasattr(controller, 'shadow_calculator'):
                controller.shadow_calculator.configure(settings)
        
        # Create log directory if it doesn't exist
        if self.enable_logging and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Initialize power history buffers
        self.power_history = deque(maxlen=300)  # 5 minutes at 60 FPS
        self.panel_power_history = {}
        
        # Initialize CSV writer
        self.csv_file = None
        self.csv_writer = None
        if self.enable_logging:
            self._init_csv_logger()
            
        # Initialize trajectory detector if available
        self.trajectory_detector = None
        try:
            from trajectory_detector import TrajectoryDetector
            self.trajectory_detector = TrajectoryDetector(controller.panel_df)
            print("Trajectory detector initialized for power-based cloud tracking")
        except ImportError:
            print("Trajectory detector not available, cloud tracking will use physics only")
            
        # Inverter parameters
        self.inverter_rated_power = 50.0  # kW
        self.inverter_efficiency = 0.97   # 97%
        self.inverter_threshold = 0.05    # 50W minimum power
    
    def _init_csv_logger(self):
        """Initialize CSV logging"""
        try:
            csv_path = os.path.join(self.log_dir, "panel_power.csv")
            
            # Check if file exists to determine if header is needed
            file_exists = os.path.isfile(csv_path)
            
            self.csv_file = open(csv_path, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header if new file
            if not file_exists:
                header = ['timestamp', 'simulation_time', 'panel_id', 'power_kw', 
                         'baseline_kw', 'coverage', 'ac_power']
                self.csv_writer.writerow(header)
                print(f"Created new power log at {csv_path}")
            else:
                print(f"Appending to existing power log at {csv_path}")
        except Exception as e:
            print(f"Error initializing CSV logger: {e}")
            self.enable_logging = False
    
    def __del__(self):
        """Cleanup resources on deletion"""
        if self.csv_file:
            try:
                self.csv_file.close()
            except:
                pass
    
    def update(self, result, timestamp):
        """
        Update power statistics and logging.
        
        Args:
            result: Simulation result dictionary
            timestamp: Current timestamp
        """
        # Safety check for result and power_output
        if not isinstance(result, dict):
            result = {'power_output': {'total': 0.0, 'total_ac': 0.0}}
        
        power_output = result.get('power_output', {})
        
        # Ensure power_output is a dictionary
        if not isinstance(power_output, dict):
            power_output = {'total': 0.0, 'total_ac': 0.0}
        
        # Apply inverter clipping and efficiency
        power_output = self._apply_inverter_effects(power_output)
        
        # Store power history
        if 'total_ac' in power_output:
            self.power_history.append(power_output['total_ac'])
        
        # Log power data
        if self.enable_logging and self.csv_writer:
            self._log_power_data(power_output, timestamp, self.controller.current_hour)
        
        # Update trajectory detector
        if self.trajectory_detector:
            try:
                panel_power_dict = {}
                for panel_id, data in power_output.items():
                    if panel_id not in ('total', 'total_ac', 'baseline_total', 'farm_reduction_pct'):
                        # Ensure data is a dictionary
                        if isinstance(data, dict):
                            final_power = data.get('final_power', 0)
                        else:
                            # If data is not a dictionary, use it directly
                            final_power = float(data) if data is not None else 0.0
                        panel_power_dict[panel_id] = final_power
                
                # Update trajectory from power data
                trajectory_result = self.trajectory_detector.update_from_power(
                    panel_power_dict, timestamp
                )
                
                # Only update trajectory in the result if confidence is good enough
                if (trajectory_result and 
                    trajectory_result.get('confidence', 0) > 0.4 and
                    'cloud_speed' in result and result['cloud_speed'] is None):
                    # Update result with trajectory information
                    result['cloud_speed'] = trajectory_result.get('speed')
                    result['cloud_direction'] = trajectory_result.get('direction')
                    result['confidence'] = trajectory_result.get('confidence')
                    result['trajectory_source'] = 'power'
            except Exception as e:
                print(f"Error in trajectory detection: {e}")
        
        return result
    
    def _apply_inverter_effects(self, power_output):
        """Apply inverter efficiency, clipping, and shutdown threshold"""
        # Get total DC power
        try:
            if isinstance(power_output, dict):
                total_dc = power_output.get('total', 0.0)
            else:
                total_dc = float(power_output) if power_output is not None else 0.0
                power_output = {'total': total_dc}
        except:
            total_dc = 0.0
            power_output = {'total': 0.0}
        
        # Apply inverter efficiency
        total_ac_unclipped = total_dc * self.inverter_efficiency
        
        # Apply inverter clipping
        total_ac = min(total_ac_unclipped, self.inverter_rated_power)
        
        # Apply inverter shutdown threshold
        if total_ac < self.inverter_threshold:
            total_ac = 0.0
        
        # Add AC values to output
        power_output['total_ac'] = total_ac
        power_output['inverter_clipping'] = max(0.0, total_ac_unclipped - total_ac)
        power_output['inverter_efficiency_loss'] = total_dc - total_ac_unclipped
        
        return power_output
    
    def _log_power_data(self, power_output, timestamp, simulation_time):
        """
        Log power data to CSV.
        
        Args:
            power_output: Dictionary with power output information
            timestamp: Current timestamp
            simulation_time: Current simulation time (hours)
        """
        try:
            current_time = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Log each panel's power
            for panel_id, data in power_output.items():
                if panel_id in ('total', 'total_ac', 'baseline_total', 'farm_reduction_pct', 
                              'inverter_clipping', 'inverter_efficiency_loss'):
                    continue
                
                # Ensure data is a dictionary
                if isinstance(data, dict):
                    power = data.get('final_power', 0)
                    baseline = data.get('baseline', 0)
                    coverage = data.get('coverage', 0)
                else:
                    # If data is not a dictionary, use it directly
                    power = float(data) if data is not None else 0.0
                    baseline = 0.0
                    coverage = 0.0
                
                row = [current_time, simulation_time, panel_id, power, baseline, coverage, power_output.get('total_ac', 0)]
                self.csv_writer.writerow(row)
            
            # Flush periodically
            if self.controller.frame_count % 60 == 0:
                self.csv_file.flush()
        except Exception as e:
            print(f"Error logging power data: {e}")
    
    def get_power_stats(self):
        """
        Get power statistics.
        
        Returns:
            Dictionary with power statistics
        """
        if not self.power_history:
            return {
                'current': 0.0,
                'avg_1min': 0.0,
                'avg_5min': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        # Current power
        current = self.power_history[-1]
        
        # 1-minute average (60 samples at 60 FPS)
        samples_1min = min(60, len(self.power_history))
        avg_1min = sum(list(self.power_history)[-samples_1min:]) / samples_1min
        
        # 5-minute average (300 samples at 60 FPS)
        avg_5min = sum(self.power_history) / len(self.power_history)
        
        # Min and max
        min_power = min(self.power_history)
        max_power = max(self.power_history)
        
        return {
            'current': current,
            'avg_1min': avg_1min,
            'avg_5min': avg_5min,
            'min': min_power,
            'max': max_power
        }

def integrate_with_simulation_controller(controller):
    """
    Integrate power calculations with simulation controller.
    
    Args:
        controller: SimulationController instance
    
    Returns:
        Updated controller
    """
    # Import settings
    try:
        from cloud_simulation import Settings
    except ImportError:
        Settings = type('Settings', (), {
            'CLOUD_TRANSMITTANCE': 0.2,
            'SHADOW_FADE_MS': 500,
            'LOG_DIR': "./logs",
            'ENABLE_POWER_LOGGING': True
        })
    
    # Create power integration
    power_integration = PowerIntegration(controller, Settings)
    
    # Store original step method
    original_step = controller.step
    
    # Override step method to include power calculations
    def enhanced_step():
        try:
            # Run original step
            result = original_step()
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {
                    'time': datetime.now(),
                    'cloud_ellipses': [],
                    'panel_coverage': {},
                    'power_output': {'total': 0.0, 'baseline_total': 0.0, 'farm_reduction_pct': 0.0},
                    'total_power': 0.0,
                    'cloud_cover': 0,
                    'cloud_speed': None,
                    'cloud_direction': None,
                    'confidence': 0
                }
            
            # Calculate power reduction if not already done
            if 'power_output' not in result and 'panel_coverage' in result:
                panel_coverage = result['panel_coverage']
                power_output = controller.shadow_calculator.calculate_power_reduction(
                    panel_coverage, controller.panel_df
                )
                result['power_output'] = power_output
            
            # Update power statistics and logging
            result = power_integration.update(result, result.get('time', datetime.now()))
            
            return result
        except Exception as e:
            print(f"Error in enhanced step: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a basic result as fallback
            return {
                'time': datetime.now(),
                'cloud_ellipses': [],
                'panel_coverage': {},
                'power_output': {'total': 0.0, 'baseline_total': 0.0, 'farm_reduction_pct': 0.0},
                'total_power': 0.0,
                'cloud_cover': 0,
                'cloud_speed': None,
                'cloud_direction': None,
                'confidence': 0
            }
    
    # Replace step method
    controller.step = enhanced_step
    
    # Add power integration to controller
    controller.power_integration = power_integration
    
    return controller