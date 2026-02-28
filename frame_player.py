import os, glob, re, cv2, torch, queue, threading
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
FRAMES_FOLDER = "data/test/" 
CHECKPOINT_PATH = "model/best_resnet_model.pth"
CONFIDENCE_THRESHOLD = 0.80   
PIXEL_THRESHOLD = 2500        
REQUIRED_FRAMES = 5           
COOLDOWN_FRAMES = 60          

# --- MULTITHREADING SETUP ---
inference_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

def ai_worker(model, device, transform):
    """Background thread for heavy ResNet math"""
    while True:
        flow_image = inference_queue.get()
        if flow_image is None: break
        
        input_tensor = transform(Image.fromarray(cv2.cvtColor(flow_image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(input_tensor).item()
        
        if result_queue.full(): result_queue.get()
        result_queue.put(prob)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256), 
        nn.ReLU(), 
        nn.Dropout(0.5), 
        nn.Linear(256, 1), 
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Start Thread
    threading.Thread(target=ai_worker, args=(model, device, transform), daemon=True).start()

    # 3. Load Frames
    frame_files = sorted(glob.glob(os.path.join(FRAMES_FOLDER, '*.jpg')), 
                         key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))

    # --- INITIALIZE BUFFERS (Using 480x270 for Speed) ---
    first_frame = cv2.resize(cv2.imread(frame_files[0]), (480, 270))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros((270, 480, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    prev_flow = None
    trigger_counter, cooldown_counter = 0, 0
    is_alarm_active = False
    ai_probability = 0.0 
    bright_pixels = 0 
    flow_bgr = np.zeros((270, 480, 3), dtype=np.uint8)

    for i in range(1, len(frame_files)):
        img = cv2.imread(frame_files[i])
        if img is None: continue
        
        curr_frame = cv2.resize(img, (480, 270))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # --- PHASE A: FAST OPTICAL FLOW (Optimized for 60 FPS) ---
        # Lowered iterations to 3 and adjusted poly_n/sigma for speed
        curr_flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        if prev_flow is not None:
            flow_diff = curr_flow - prev_flow
            magnitude, angle = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
            
            # FIXED: Perspective Trap (240 instead of 320 because height is only 270)
            magnitude[240:, :] = 0 
            magnitude[magnitude < 3.0] = 0.0 
            
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = np.clip(magnitude * 10.0, 0, 255).astype(np.uint8)
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Push to AI thread
            if not inference_queue.full():
                inference_queue.put(flow_bgr)
            
            # Pull result if ready
            if not result_queue.empty():
                ai_probability = result_queue.get()

            bright_pixels = np.sum(hsv[..., 2] > 50)
            
            # Accident Trigger Logic
            instant_hit = (ai_probability > CONFIDENCE_THRESHOLD) and (bright_pixels > PIXEL_THRESHOLD)
            if instant_hit:
                trigger_counter += 1
            else:
                trigger_counter = max(0, trigger_counter - 1)

            if trigger_counter >= REQUIRED_FRAMES:
                is_alarm_active = True
                cooldown_counter = COOLDOWN_FRAMES

        # Cooldown Logic
        if is_alarm_active:
            cooldown_counter -= 1
            if cooldown_counter <= 0:
                is_alarm_active = False
                trigger_counter = 0

        # --- DRAWING ---
        display_frame = curr_frame.copy()
        text_color = (0, 0, 255) if is_alarm_active else (0, 255, 0)
        status = "!! ACCIDENT !!" if is_alarm_active else "Safe"
        
        cv2.putText(display_frame, f"STATUS: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(display_frame, f"AI Prob: {ai_probability:.2f} | Px: {bright_pixels}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if is_alarm_active: 
            cv2.rectangle(display_frame, (0,0), (480,270), (0,0,255), 10)

        # Stitch and Display
        cv2.imshow("Live Frame Player Engine", np.hstack((display_frame, flow_bgr)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        # Prepare for next frame
        prev_gray = curr_gray
        prev_flow = curr_flow

    cv2.destroyAllWindows()

if __name__ == '__main__': 
    main()