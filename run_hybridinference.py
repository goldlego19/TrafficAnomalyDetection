import os, glob, re, cv2, torch, queue, threading
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image

# --- IMPORT YOUR MODULAR FILES ---
from config import TrafficConfig

# --- CONFIGURATION ---
FRAMES_FOLDER = "data/marsaBridge/" 
YOLO_MODEL_PATH = "yolo11n.pt" 
CHECKPOINT_PATH = "model/best_resnet_model.pth"

# --- AI THRESHOLDS ---
CONF_THRESH = 0.80
PIXEL_THRESH = 2500
REQUIRED_FRAMES = 10  
COOLDOWN_FRAMES = 120

# --- MULTITHREADING QUEUES ---
inference_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

def ai_worker(model, device, transform):
    while True:
        flow_image = inference_queue.get()
        if flow_image is None: break
        input_tensor = transform(Image.fromarray(cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(input_tensor).item()
        if result_queue.full(): result_queue.get()
        result_queue.put(prob)

def build_lane_masks(frame_shape, config):
    lane_masks = {}
    for lane_name, poly in config.lane_polygons.items():
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        lane_masks[lane_name] = mask
    return lane_masks

def get_transform_matrix(config):
    dst_points = np.float32([[0, 0], [config.real_width_m, 0], 
                            [config.real_width_m, config.real_length_m], [0, config.real_length_m]])
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

# --- ANTI-SHAKE STABILISER ---
class StaticCameraStabiliser:
    def __init__(self, reference_frame):
        self.ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.ref_kpts, self.ref_desc = self.orb.detectAndCompute(self.ref_gray, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def stabilise(self, frame):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_kpts, curr_desc = self.orb.detectAndCompute(curr_gray, None)

        if self.ref_desc is None or curr_desc is None:
            return frame

        matches = self.matcher.match(curr_desc, self.ref_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50] 

        if len(good_matches) < 10:
            return frame

        src_pts = np.float32([curr_kpts[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.ref_kpts[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)

        if matrix is not None:
            h, w = frame.shape[:2]
            return cv2.warpAffine(frame, matrix, (w, h))
        return frame

def main():
    config = TrafficConfig("marsaBridge.mp4") 
    config.load()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Sequential(nn.Linear(resnet.fc.in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1), nn.Sigmoid())
    resnet.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    resnet.to(device).eval()

    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    threading.Thread(target=ai_worker, args=(resnet, device, resnet_transform), daemon=True).start()
    yolo_model = YOLO(YOLO_MODEL_PATH)

    frame_files = sorted(glob.glob(os.path.join(FRAMES_FOLDER, '*.jpg')), 
                         key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))

    first_frame = cv2.imread(frame_files[0])
    stabiliser = StaticCameraStabiliser(first_frame)
    
    transform_matrix = get_transform_matrix(config)
    lane_masks = build_lane_masks(first_frame.shape, config)
    
    prev_gray = cv2.cvtColor(cv2.resize(first_frame, (480, 270)), cv2.COLOR_BGR2GRAY)
    hsv_buffer = np.zeros((270, 480, 3), dtype=np.uint8)
    hsv_buffer[..., 1] = 255
    
    track_history = defaultdict(list)
    kalman_filters = {}
    frame_count, trigger_counter, cooldown_counter = 0, 0, 0
    ai_probability, bright_pixels = 0.0, 0
    is_accident_active = False
    flow_view = np.zeros((270, 480, 3), dtype=np.uint8)

    while True:
        for i in range(1, len(frame_files)):
            raw_frame = cv2.imread(frame_files[i])
            if raw_frame is None: continue
            
            frame = stabiliser.stabilise(raw_frame)
            frame_count += 1

            # --- BRANCH A: PHYSICS ---
            curr_gray = cv2.cvtColor(cv2.resize(frame, (480, 270)), cv2.COLOR_BGR2GRAY)
            curr_flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 3, 3, 5, 1.1, 0)
            mag, ang = cv2.cartToPolar(curr_flow[..., 0], curr_flow[..., 1])
            mag[240:, :] = 0; mag[mag < 3.0] = 0.0 
            hsv_buffer[..., 0] = ang * 180 / np.pi / 2
            hsv_buffer[..., 2] = np.clip(mag * 10.0, 0, 255).astype(np.uint8)
            flow_view = cv2.cvtColor(hsv_buffer, cv2.COLOR_HSV2BGR)

            if not inference_queue.full(): inference_queue.put(flow_view)
            if not result_queue.empty(): ai_probability = result_queue.get()

            bright_pixels = np.sum(hsv_buffer[..., 2] > 50)
            if (ai_probability > CONF_THRESH) and (bright_pixels > PIXEL_THRESH):
                trigger_counter += 1
            else: trigger_counter = max(0, trigger_counter - 1)

            if trigger_counter >= REQUIRED_FRAMES:
                is_accident_active, cooldown_counter = True, COOLDOWN_FRAMES
            
            if is_accident_active:
                cooldown_counter -= 1
                if cooldown_counter <= 0: is_accident_active, trigger_counter = False, 0

            # --- BRANCH B: TRACKING & LANE AVERAGES ---
            results = yolo_model.track(frame, persist=True, classes=[2, 5, 7], conf=0.15, verbose=False)
            current_lane_speeds = defaultdict(list)
            
            overlay = frame.copy()
            for lane_name, poly in config.lane_polygons.items():
                cv2.fillPoly(overlay, [poly], (255, 255, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    # --- 1.PRIVACY BLUR ---
                    h, w = frame.shape[:2]
                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(w, x2), min(h, y2)
                    
                    if y2_c > y1_c and x2_c > x1_c:
                        car_roi = frame[y1_c:y2_c, x1_c:x2_c]
                        blurred_car = cv2.GaussianBlur(car_roi, (51, 51), 0) 
                        frame[y1_c:y2_c, x1_c:x2_c] = blurred_car
                    
                    cx, cy = (x1 + x2) // 2, y2 - 5
                    
                    assigned_lane = "Unknown"
                    for lane_name, mask in lane_masks.items():
                        if mask[cy, cx] == 255: assigned_lane = lane_name; break

                    if assigned_lane == "Unknown":
                        continue

                    pt = np.array([[[cx, cy]]], dtype=np.float32)
                    warped = cv2.perspectiveTransform(pt, transform_matrix)[0][0]
                    track_history[track_id].append((warped[0], warped[1], frame_count))
                    
                    if len(track_history[track_id]) > 10:
                        o_x, o_y, o_f = track_history[track_id][0]
                        dist = np.sqrt((warped[0]-o_x)**2 + (warped[1]-o_y)**2)
                        time_sec = (frame_count - o_f) / 60.0
                        speed_kmh = (dist / time_sec) * 3.6 if time_sec > 0 else 0

                        if track_id not in kalman_filters: kalman_filters[track_id] = SpeedKalmanFilter()
                        smooth_speed = int(max(0, kalman_filters[track_id].update(speed_kmh)))
                        
                        if smooth_speed > 5:
                            current_lane_speeds[assigned_lane].append(smooth_speed)

                        color = (0, 0, 255) if is_accident_active else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID:{track_id} | {smooth_speed} km/h", (x1, y1-10), 0, 0.5, color, 2)

            # --- HUD ASSEMBLY ---
            main_w, main_h = 1080, 800 
            side_w = 400
            canvas = np.zeros((main_h, main_w + side_w, 3), dtype=np.uint8)
            
            canvas[:, :main_w] = cv2.resize(frame, (main_w, main_h))
            
            stats_panel = canvas[:, main_w:]
            flow_small = cv2.resize(flow_view, (side_w, 280)) 
            stats_panel[:280, :] = flow_small
            
            cv2.putText(stats_panel, "LANE PERFORMANCE", (20, 330), 0, 0.8, (255, 255, 255), 2)
            cv2.line(stats_panel, (20, 345), (side_w-20, 345), (100, 100, 100), 1)

            y_offset = 390
            for lane_name in sorted(config.lane_polygons.keys()):
                speeds = current_lane_speeds.get(lane_name, [])
                avg = int(sum(speeds) / len(speeds)) if speeds else 0
                count = len(speeds)

                # --- TRAFFIC STATUS FILTER ---
                if count == 0:
                    status_text = "Empty"
                    status_color = (200, 200, 200) # Light Grey
                elif avg <= 10:
                    status_text = "Severe Traffic"
                    status_color = (0, 0, 255)     # Red
                elif avg <= 20:
                    status_text = "Moderate"
                    status_color = (0, 165, 255)   # Orange
                else:
                    status_text = "Good"
                    status_color = (0, 255, 0)     # Green

                # Draw the lane name and numbers
                cv2.putText(stats_panel, f"{lane_name}: {avg} km/h ({count} cars)", (20, y_offset), 0, 0.6, (255, 255, 255), 1)
                
                # Draw the colour-coded status next to it
                cv2.putText(stats_panel, f"[{status_text}]", (280, y_offset), 0, 0.6, status_color, 2)
                
                y_offset += 50

            cv2.putText(stats_panel, "ANOMALY AI", (20, 680), 0, 0.8, (255, 255, 255), 2)
            cv2.line(stats_panel, (20, 695), (side_w-20, 695), (100, 100, 100), 1)
            
            ai_color = (0, 0, 255) if is_accident_active else (0, 255, 0)
            cv2.putText(stats_panel, f"Prob: {ai_probability:.2f}", (20, 735), 0, 0.6, ai_color, 1)
            cv2.putText(stats_panel, f"Pixels: {bright_pixels}", (20, 770), 0, 0.6, ai_color, 1)

            if is_accident_active:
                cv2.rectangle(canvas, (0,0), (main_w, main_h), (0,0,255), 15)
                cv2.putText(canvas, "ACCIDENT ALERT", (350, 400), 2, 1.5, (0,0,255), 3)

            cv2.imshow("Hybrid Tall Dashboard", canvas)
            prev_gray = curr_gray
            if cv2.waitKey(1) & 0xFF == ord('q'): return

if __name__ == '__main__': 
    main()