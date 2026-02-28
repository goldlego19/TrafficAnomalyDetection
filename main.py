import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# --- IMPORT OUR MODULAR FILES ---
from config import TrafficConfig
from calibration import calibrate_physics, calibrate_lanes

# --- CONFIGURATION ---
VIDEO_PATH = "videos/testcropped.mp4" # Change this to your tracking video
YOLO_MODEL_PATH = "yolo11n.pt" 
FPS = 60
EFFECTIVE_FPS = FPS /2

def build_lane_masks(frame_shape, config):
    lane_masks = {}
    for lane_name, poly in config.lane_polygons.items():
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        lane_masks[lane_name] = mask
    return lane_masks

def get_transform_matrix(config):
    dst_points = np.float32([
        [0, 0], [config.real_width_m, 0], 
        [config.real_width_m, config.real_length_m], [0, config.real_length_m]
    ])
    return cv2.getPerspectiveTransform(config.src_points, dst_points)

class SpeedKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf.processNoiseCov = np.array([[1e-2, 0], [0, 1e-2]], np.float32)
        self.kf.measurementNoiseCov = np.array([[0.5]], np.float32)
        self.kf.errorCovPost = np.array([[1, 0], [0, 1]], np.float32)
        self.initialized = False

    def update(self, measured_speed):
        if not self.initialized:
            self.kf.statePost = np.array([[np.float32(measured_speed)], [0.0]], np.float32)
            self.initialized = True
        self.kf.predict()
        estimated = self.kf.correct(np.array([[np.float32(measured_speed)]]))
        return float(estimated[0][0])

def main():
    config = TrafficConfig(VIDEO_PATH)
    config.load()
    transform_matrix = get_transform_matrix(config)
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    success, initial_frame = cap.read()
    if not success: return
    
    lane_masks = build_lane_masks(initial_frame.shape, config)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    track_history = defaultdict(list)
    vehicle_speeds = {}
    kalman_filters = {}
    stationary_timers = defaultdict(int) 
    frame_count = 0
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)] 

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1

        results = yolo_model.track(frame, persist=True, classes=[2, 5, 7], conf=0.15, imgsz=736, verbose=False)

        overlay = frame.copy()
        cv2.polylines(overlay, [np.int32(config.src_points)], True, (255, 0, 255), 2)
        for idx, (lane_name, poly) in enumerate(config.lane_polygons.items()):
            col = colours[idx % len(colours)]
            cv2.fillPoly(overlay, [poly], col)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        current_lane_speeds = defaultdict(list)
        active_vehicles = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                assigned_lane = "Unknown"
                bbox_area = (x2 - x1) * (y2 - y1)
                for lane_name, mask in lane_masks.items():
                    overlap = cv2.countNonZero(mask[y1:y2, x1:x2])
                    if overlap > (bbox_area * 0.15):
                        assigned_lane = lane_name; break

                if assigned_lane == "Unknown": continue 

                cx = int((x1 + x2) / 2)
                cy = y2 - int(0.05 * (y2 - y1))
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                warped_pt = cv2.perspectiveTransform(pt, transform_matrix)[0][0]

                track_history[track_id].append((warped_pt[0], warped_pt[1], frame_count))
                if len(track_history[track_id]) > EFFECTIVE_FPS: track_history[track_id].pop(0)

                speed_kmh = 0
                if len(track_history[track_id]) >= int(EFFECTIVE_FPS / 2):
                    old_x, old_y, old_f = track_history[track_id][0]
                    new_x, new_y, new_f = track_history[track_id][-1]
                    time_seconds = (new_f - old_f) / EFFECTIVE_FPS
                    if time_seconds > 0:
                        distance_m = np.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2)
                        speed_kmh = (distance_m / time_seconds) * 3.6

                if track_id not in kalman_filters: kalman_filters[track_id] = SpeedKalmanFilter()
                
                smoothed_speed = kalman_filters[track_id].update(speed_kmh)
                display_speed = int(max(0, smoothed_speed)) 
                vehicle_speeds[track_id] = display_speed
                current_lane_speeds[assigned_lane].append(display_speed)

                active_vehicles.append({'id': track_id, 'box': (x1, y1, x2, y2), 'speed': display_speed, 'lane': assigned_lane})

        lane_averages = {l: (sum(s for s in sp if s > 5) / len([s for s in sp if s > 5]) if [s for s in sp if s > 5] else 0) for l, sp in current_lane_speeds.items()}

        for v in active_vehicles:
            x1, y1, x2, y2 = v['box']
            color, label = (0, 255, 0), f"ID:{v['id']} {v['speed']}km/h"
            
            if v['lane'] != "Unknown" and v['speed'] < 3 and lane_averages[v['lane']] > 15:
                stationary_timers[v['id']] += 1
            else: stationary_timers[v['id']] = 0

            if stationary_timers[v['id']] >= (EFFECTIVE_FPS * 5):
                color, label = (0, 0, 255), f"🚨 ANOMALY! ID:{v['id']}"
                cv2.putText(frame, "TRAFFIC ANOMALY DETECTED", (50, 70), 0, 1.2, (0, 0, 255), 3)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 0, 0.6, color, 2)

        cv2.imshow("YOLO Tracking & Speed Engine", frame)
        
        key = cv2.waitKey(16) & 0xFF
        if key == ord("q"): break
        elif key == ord("p"):
            print("\n⏸️ PAUSED. Select: [f] Full, [s] Square, [l] Lanes, [p] Resume")
            while True:
                pause_key = cv2.waitKey(0) & 0xFF
                if pause_key == ord("p"): break
                elif pause_key in [ord("f"), ord("s"), ord("l")]:
                    cv2.destroyWindow("YOLO Tracking & Speed Engine")
                    cv2.waitKey(1) 
                    if pause_key == ord("f"):
                        calibrate_physics(frame, config)
                        calibrate_lanes(frame, config)
                    elif pause_key == ord("s"): calibrate_physics(frame, config)
                    elif pause_key == ord("l"): calibrate_lanes(frame, config)
                    transform_matrix = get_transform_matrix(config)
                    lane_masks = build_lane_masks(frame.shape, config)
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__': main()