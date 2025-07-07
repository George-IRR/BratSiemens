# pi_sender.py - Modified to send raw images
import asyncio
import base64
import json
import requests
import cv2
import time
import threading
from ultralytics import YOLO
from ImgCropDetect.aruco_cropper import ArucoCropper

API_URL = "http://172.16.55.12:5003/data"  # your laptop's IP!

# Global variable to control what to send
send_raw_mode = False

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def send_image_data(image, detections, is_raw=False):
    # Get image dimensions for crop_shape
    h, w = image.shape[:2]  # height, width from image shape
    
    payload = {
        "image": encode_image_to_base64(image),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "detections": detections if not is_raw else [],
        "crop_shape": [w, h],  # OBLIGATORIU: [width, height]
        "raw_image": is_raw  # Flag to indicate raw image
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=3)
        if response.status_code == 200:
            result = response.json()
            status = result.get('status', 'unknown')
            img_type = "RAW" if is_raw else "PROCESSED"
            print(f"📤 Sent {img_type} frame {w}x{h} with {len(detections)} detections - BLE Status: {status}")
        else:
            print(f"❌ Send failed: {response.status_code}")
    except Exception as e:
        print("❌ Failed to send:", e)

def command_listener():
    """Listen for commands to toggle raw image mode."""
    global send_raw_mode
    print("🎮 Command listener started! Type 'raw' to toggle raw image mode, 'quit' to exit")
    
    while True:
        try:
            cmd = input().strip().lower()
            if cmd == 'raw':
                send_raw_mode = not send_raw_mode
                mode = "RAW" if send_raw_mode else "PROCESSED"
                print(f"📸 Switched to {mode} mode")
            elif cmd == 'quit':
                print("🛑 Exiting...")
                break
        except (EOFError, KeyboardInterrupt):
            break
def main():
    cropper = ArucoCropper(camera_resolution=(4608, 2592), reference_ids=[0,1,2,3])
    model = YOLO("ModelV3.2.pt")
    target = ['triangle', 'rectangle', 'arch', 'cube']
    ids = [i for i, n in model.names.items() if n in target]
    
    print("� Starting detection with raw image support...")
    print("Type 'raw' to toggle raw image mode")
    
    # Start command listener in background thread
    cmd_thread = threading.Thread(target=command_listener, daemon=True)
    cmd_thread.start()
    
    while True:
        frame = cropper.capture_frame()
        
        if send_raw_mode:
            # Send raw frame without processing
            if frame is not None:
                h, w, _ = frame.shape
                print(f"📷 Sending RAW frame: {w}x{h}")
                send_image_data(frame, [], is_raw=True)
        else:
            # Normal detection mode
            cropped = cropper.get_cropped_image(frame)
            
            if cropped is None:
                continue
            
            h, w, _ = cropped.shape
            print(f"📷 Processing frame: {w}x{h}")
            
            res = model(cropped, verbose=False)
            dets = []
            
            for box in res[0].boxes:
                c = int(box.cls.item())
                if c not in ids:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                dets.append({
                    'class': model.names[c],
                    'confidence': float(box.conf.item()),
                    'center_px': [cx, h - cy]  # Note: h - cy for coordinate flip
                })
            
            if dets:
                print(f"🎯 Found {len(dets)} detections:")
                for det in dets:
                    print(f"   {det['class']}: {det['confidence']:.3f} at ({det['center_px'][0]:.0f}, {det['center_px'][1]:.0f})")
            
            send_image_data(cropped, dets)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()
