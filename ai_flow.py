import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# --- CONFIGURATION (EXACT MATCH TO YOUR BATCH SCRIPT) ---
VIDEO_PATH = "videos/testcropped.mp4" 
CHECKPOINT_PATH = "model/best_resnet_model.pth"

# --- THE DUAL-KEY THRESHOLDS (EXACT MATCH) ---
CONFIDENCE_THRESHOLD = 0.85  
PIXEL_THRESHOLD = 3000        
REQUIRED_FRAMES = 5           
COOLDOWN_FRAMES = 60          

# --- CRITICAL: FRAME SKIP SIMULATOR ---
# If you extracted your CADP frames at 30fps from a 60fps video, 
# the cars jump twice as far between frames. Set this to 2 to replicate that exact physics jump.
FRAME_SKIP = 1 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initialising Live Inference on {device}...")
    
    # 1. Load the frozen AI Brain exactly as in your script
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Open Live Video Instead of Folders
    cap = cv2.VideoCapture(VIDEO_PATH)
    success, first_frame = cap.read()
    if not success: 
        print("Failed to load video.")
        return

    # 3. Initialise the physics variables (Exactly as in your script)
    first_frame = cv2.resize(first_frame, (640, 360))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros((360, 640, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    prev_flow = None
    
    trigger_counter = 0
    cooldown_counter = 0
    is_alarm_active = False
    frame_counter = 0

    while cap.isOpened():
        success, raw_frame = cap.read()
        if not success: break
        
        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue
            
        # --- THE SECRET FIX: JPEG COMPRESSION REPLICATION ---
        # Compress the raw MP4 frame in memory to exactly match your extraction script.
        # This forces the live video to have the exact same pixel artifacts the AI was trained on.
        _, encoded_img = cv2.imencode('.jpg', raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        # Now pass the JPEG-simulated image into the standard pipeline
        curr_frame = cv2.resize(img, (640, 360))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # --- PHASE A: CALCULATE VELOCITY ---
        curr_flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                                 0.5, 3, 15, 3, 5, 1.2, 0)

        ai_probability = 0.0
        bright_pixels = 0
        flow_bgr = np.zeros((360, 640, 3), dtype=np.uint8)

        # --- PHASE B: CALCULATE ACCELERATION ---
        if prev_flow is not None:
            flow_diff = curr_flow - prev_flow
            magnitude, angle = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
            
            # --- PHASE C: PHYSICAL FILTERS ---
            magnitude[320:, :] = 0 
            magnitude[magnitude < 3.0] = 0.0 
            
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = np.clip(magnitude * 10.0, 0, 255).astype(np.uint8)
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # --- PHASE D: AI PREDICTION ---
            input_tensor = transform(Image.fromarray(cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            with torch.no_grad():
                ai_probability = model(input_tensor).item()

            # --- PHASE E: DUAL-KEY ALARM LOGIC ---
            bright_pixels = np.sum(hsv[..., 2] > 50)
            
            instant_hit = (ai_probability > CONFIDENCE_THRESHOLD) and (bright_pixels > PIXEL_THRESHOLD)

            if instant_hit:
                trigger_counter += 1
            else:
                trigger_counter = max(0, trigger_counter - 1)

            if trigger_counter >= REQUIRED_FRAMES:
                is_alarm_active = True
                cooldown_counter = COOLDOWN_FRAMES

        if is_alarm_active:
            cooldown_counter -= 1
            if cooldown_counter <= 0:
                is_alarm_active = False
                trigger_counter = 0

        # --- PHASE F: VISUALISATION ---
        display_frame = curr_frame.copy()
        text_color = (0, 0, 255) if is_alarm_active else (0, 255, 0)
        status = "!! ACCIDENT !!" if is_alarm_active else "Safe"
        
        cv2.putText(display_frame, f"STATUS: {status}", (20, 50), 2, 1.2, text_color, 3)
        cv2.putText(display_frame, f"AI Prob: {ai_probability:.2f} | Pixels: {bright_pixels}", 
                    (20, 90), 2, 0.6, (255, 255, 255), 1)

        if is_alarm_active:
            cv2.rectangle(display_frame, (0,0), (640,360), (0,0,255), 15)

        final_view = np.hstack((display_frame, flow_bgr))
        cv2.imshow("Standalone Optical Flow AI", final_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = curr_gray
        prev_flow = curr_flow

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()