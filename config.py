import os
import json
import numpy as np

class TrafficConfig:
    def __init__(self, video_path):
        # Dynamically create a clean config filename
        base_name = os.path.basename(video_path)
        self.video_name = os.path.splitext(base_name)[0] 
        
        # Route the file to the 'configs/' directory
        self.config_dir = "configs"
        self.config_file = os.path.join(self.config_dir, f"{self.video_name}_config.txt")

        # Default physics and mapping
        self.src_points = np.float32([[300, 200], [900, 200], [1100, 600], [100, 600]])
        self.lane_polygons = {}
        self.real_width_m = 3.5
        self.real_length_m = 10.0
        self.pixels_per_meter = 10

    def load(self):
        """Loads the configuration from the dynamic text file if it exists."""
        if os.path.exists(self.config_file):
            print(f"Loading saved calibration from {self.config_file}...")
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.src_points = np.float32(data['SRC_POINTS'])
                self.lane_polygons = {k: np.array(v, np.int32) for k, v in data['LANE_POLYGONS'].items()}
                self.real_width_m = data.get('REAL_WIDTH_M', 3.5)
                self.real_length_m = data.get('REAL_LENGTH_M', 10.0)

    def save(self):
        """Saves the current configuration to the dynamic text file."""
        # Ensure the configs directory actually exists before saving
        os.makedirs(self.config_dir, exist_ok=True)
        
        print(f"Saving calibration to {self.config_file}...")
        data = {
            'SRC_POINTS': self.src_points.tolist(),
            'LANE_POLYGONS': {k: v.tolist() for k, v in self.lane_polygons.items()},
            'REAL_WIDTH_M': self.real_width_m,
            'REAL_LENGTH_M': self.real_length_m
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=4)