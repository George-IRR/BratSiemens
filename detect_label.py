import cv2
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# === CONFIG ===
USE_PI_CAMERA = False
output_dir = "dataset_debug"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

classes = {
    "triangle": 0, "rectangle": 1, "arch": 2,
    "half-circle": 3, "cylinder": 4, "cube": 5
}

# Load saved settings if exists
SETTINGS_PATH = "settings.json"
if os.path.exists(SETTINGS_PATH):
    with open(SETTINGS_PATH, "r") as f:
        calib = json.load(f)
else:
    calib = {
        "blur": 5,
        "hmin": 0, "hmax": 179,
        "smin": 0, "smax": 255,
        "vmin": 0, "vmax": 255
    }

def detect_shapes_and_label(frame, hsv_ranges, blur_val):
    hmin, hmax, smin, smax, vmin, vmax = hsv_ranges
    labeled_objects = []

    blur_val = blur_val | 1
    blurred = cv2.GaussianBlur(frame, (blur_val, blur_val), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (hmin, smin, vmin), (hmax, smax, vmax))

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        vertices = len(approx)
        x, y, w, h = cv2.boundingRect(contour)

        shape_label = "unidentified"
        if vertices == 3:
            shape_label = "triangle"
        elif vertices == 4:
            aspect_ratio = w / float(h)
            shape_label = "cube" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
        elif vertices == 5:
            shape_label = "arch"
        else:
            if not cv2.isContourConvex(approx):
                shape_label = "arch"
            else:
                fill_ratio = area / float(w * h)
                shape_label = "half-circle" if fill_ratio < 0.7 else "cylinder"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, shape_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        H, W = frame.shape[:2]
        cx, cy = x + w/2, y + h/2
        labeled_objects.append((classes[shape_label], cx/W, cy/H, w/W, h/H))

    return frame, labeled_objects

def save_frame_and_labels(frame, labels):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(output_dir, "images", f"{timestamp}.jpg")
    label_path = os.path.join(output_dir, "labels", f"{timestamp}.txt")
    cv2.imwrite(img_path, frame)
    with open(label_path, "w") as f:
        for cls_id, cx, cy, w, h in labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    print(f"[SALVAT] {img_path} + {label_path}")

# === UI ===
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("[Eroare] Nu am putut accesa camera.")
    exit()

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.45)
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis("off")
fig.canvas.manager.set_window_title("Live Debug Shape Detection")

slider_defs = {
    "blur":     [0.1, 30, calib["blur"]],
    "hmin":     [0, 179, calib["hmin"]],
    "hmax":     [0, 179, calib["hmax"]],
    "smin":     [0, 255, calib["smin"]],
    "smax":     [0, 255, calib["smax"]],
    "vmin":     [0, 255, calib["vmin"]],
    "vmax":     [0, 255, calib["vmax"]],
}
sliders = {}
from matplotlib.widgets import Button
for s in sliders.values():
    s.on_changed(update_fig)
# === Buton "Preview Image" ===
ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])  # [x, y, width, height]
btn = Button(ax_button, "Preview Image")

def on_click(event):
    _, frame = cap.read()
    if frame is None: return

    params = [int(sliders[k].val) for k in ["hmin", "hmax", "smin", "smax", "vmin", "vmax"]]
    blur = int(sliders["blur"].val)
    processed, _ = detect_shapes_and_label(frame.copy(), params, blur)
    
    path = os.path.join(output_dir, "preview.jpg")
    cv2.imwrite(path, processed)
    print(f"[PREVIEW] Imagine salvată în: {path}")

btn.on_clicked(on_click)

for i, (key, (vmin, vmax, val)) in enumerate(slider_defs.items()):
    ax_slider = plt.axes([0.15, 0.4 - 0.04*i, 0.7, 0.03])
    sliders[key] = Slider(ax_slider, key, vmin, vmax, valinit=val, valstep=1)

def update_fig(val=None):
    _, frame = cap.read()
    if frame is None: return

    params = [int(sliders[k].val) for k in ["hmin", "hmax", "smin", "smax", "vmin", "vmax"]]
    blur = int(sliders["blur"].val)
    processed, labels = detect_shapes_and_label(frame.copy(), params, blur)

    im.set_data(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()

for s in sliders.values():
    s.on_changed(update_fig)

def on_key(event):
    if event.key == "c":
        _, frame = cap.read()
        params = [int(sliders[k].val) for k in ["hmin", "hmax", "smin", "smax", "vmin", "vmax"]]
        blur = int(sliders["blur"].val)
        processed, labels = detect_shapes_and_label(frame.copy(), params, blur)
        save_frame_and_labels(frame, labels)
    elif event.key == "q":
        with open(SETTINGS_PATH, "w") as f:
            json.dump({k: int(s.val) for k, s in sliders.items()}, f, indent=2)
        plt.close()
        cap.release()
        print("Ieșit și setările salvate.")

fig.canvas.mpl_connect("key_press_event", on_key)

# Loop update
import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, lambda i: update_fig(), interval=100)
plt.show()
