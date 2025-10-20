import cv2
from ultralytics import YOLO
import torch
import time
import numpy as np

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8x.pt')
model.to(device)

# Create a dummy frame (or load one from video)
frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warmup
for _ in range(10):
    _ = model.predict(source=frame, device=device, conf=0.5, imgsz=640, verbose=False, half=True)

# Benchmark pure inference
num_frames = 500
start = time.time()

for i in range(num_frames):
    with torch.amp.autocast(device_type='cuda', enabled=True):
        results = model.predict(source=frame, device=device, conf=0.5, imgsz=640, verbose=False, half=True)

end = time.time()
total_time = end - start
avg_ms = (total_time / num_frames) * 1000
fps = num_frames / total_time

print(f"\n=== Pure GPU Inference Benchmark ===")
print(f"Frames processed: {num_frames}")
print(f"Total time: {total_time:.2f}s")
print(f"Average inference: {avg_ms:.2f}ms")
print(f"Pure GPU FPS: {fps:.0f}")
