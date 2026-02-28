import cv2
import numpy as np

def get_4_points_from_user(prompt_text, base_frame):
    points = []
    clone = base_frame.copy()
    cv2.putText(clone, prompt_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.imshow("Anomaly & Speed Engine", clone)
    
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(clone, (x, y), 5, (255, 0, 255), -1)
            if len(points) > 1:
                cv2.line(clone, tuple(points[-2]), tuple(points[-1]), (255, 0, 255), 2)
            cv2.imshow("Anomaly & Speed Engine", clone)

    cv2.setMouseCallback("Anomaly & Speed Engine", click_event)
    while len(points) < 4: cv2.waitKey(10)
    return np.float32(points)

def get_polygon_from_user(prompt_text, base_frame):
    points = []
    clone = base_frame.copy()
    finished = False
    cv2.putText(clone, prompt_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(clone, "Left-Click: Add point | Right-Click: Close shape & finish", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.imshow("Anomaly & Speed Engine", clone)
    
    def click_event(event, x, y, flags, params):
        nonlocal finished
        if event == cv2.EVENT_LBUTTONDOWN and not finished:
            points.append([x, y])
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(clone, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            cv2.imshow("Anomaly & Speed Engine", clone)
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= 3:
            cv2.line(clone, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
            cv2.imshow("Anomaly & Speed Engine", clone)
            finished = True

    cv2.setMouseCallback("Anomaly & Speed Engine", click_event)
    while not finished: cv2.waitKey(10)
    return points

def calibrate_physics(frame, config):
    print("\n📐 Calibrating Physics (Perspective Square)...")
    config.src_points = get_4_points_from_user("Click 4 points for PERSPECTIVE AREA (TL, TR, BR, BL)", frame)
    
    try:
        w_input = input("\n📏 Enter REAL-WORLD WIDTH (in metres, e.g., 3.5): ")
        config.real_width_m = float(w_input)
        l_input = input("📏 Enter REAL-WORLD LENGTH (in metres, e.g., 10.0): ")
        config.real_length_m = float(l_input)
        print(f"📐 Perspective calibrated to {config.real_width_m}m wide x {config.real_length_m}m long.")
    except ValueError:
        print("⚠️ Invalid input! Defaulting to 3.5m width x 10.0m length.")
        config.real_width_m, config.real_length_m = 3.5, 10.0
    
    cv2.setMouseCallback("Anomaly & Speed Engine", lambda *args: None)
    config.save()
    print("Physics Calibration complete.")

def calibrate_lanes(frame, config):
    print("\nCalibrating Lanes...")
    config.lane_polygons.clear()
    lane_count = 1
    
    while True:
        pts = get_polygon_from_user(f"LANE {lane_count}: Outline the lane", frame)
        config.lane_polygons[f"Lane {lane_count}"] = np.array(pts, np.int32)
        add_more = input(f"Lane {lane_count} saved! Add another lane? (y/n): ").strip().lower()
        if add_more != 'y': break
        lane_count += 1
    
    cv2.setMouseCallback("Anomaly & Speed Engine", lambda *args: None)
    config.save()
    print("Lane Calibration complete.")