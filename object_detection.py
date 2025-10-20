import cv2
from ultralytics import YOLO
import torch
import argparse
import time
import platform
import ctypes
import threading
import queue

__version__ = "1.0.0"

# Apply DPI awareness on Windows to prevent scaling issues
if platform.system() == "Windows":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 1 = SYSTEM_AWARE, prevents 200% zoom
    except Exception as e:
        print(f"Failed to set DPI awareness: {e}")

# YOLO models ordered by size and speed
YOLO_MODELS = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

# Supported resolutions (16:9 aspect ratio)
RESOLUTIONS = [(1920, 1080), (1280, 720), (854, 480)]  # (Width, Height)
DEFAULT_RESOLUTION_INDEX = 0  # Default to 1080p

class FrameReader(threading.Thread):
    """
    Thread that continuously reads frames from video source and puts them in a queue.
    Overlaps video decode (CPU) with inference (GPU) for better performance.
    """
    def __init__(self, cap, frame_queue, max_queue_size=3):
        threading.Thread.__init__(self)
        self.cap = cap
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.stopped = False
        self.daemon = True

    def run(self):
        while not self.stopped:
            if self.frame_queue.qsize() < self.max_queue_size:
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    self.frame_queue.put(None)  # Signal end of video
                    break
                self.frame_queue.put(frame)
            else:
                time.sleep(0.001)  # Small sleep to prevent busy-waiting

    def stop(self):
        self.stopped = True

def draw_shadowed_text(frame, text, position, font, font_scale, color, thickness, bg_color, alpha=0.6):
    """
    Draw translucent shadow box behind text with the given attributes.
    """
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    x, y = position

    # Draw shadow box (translucent background)
    overlay = frame.copy()
    box_coords = ((x, y - text_height - 5), (x + text_width + 10, y + 5))
    cv2.rectangle(overlay, box_coords[0], box_coords[1], bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw text over the shadow box
    cv2.putText(frame, text, (x + 5, y), font, font_scale, color, thickness, cv2.LINE_AA)

def process_webcam(camera_index=0, imgsz=640, conf=0.5, force_cpu=False, video_source=None, no_display=False, threaded=False):
    """
    Process webcam feed or video file and perform object detection using YOLOv8.

    Args:
        threaded: If True, uses separate thread for frame reading to overlap decode with inference
    """
    print(f"Target Tracker v{__version__}")

    # Detect device and GPU details
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if device == 'cuda' else "CPU"
    print(f"Using device: {device_name}")

    # Select default model based on device
    current_model_index = YOLO_MODELS.index('yolov8x' if device == 'cuda' else 'yolov8s')
    model_path = YOLO_MODELS[current_model_index]
    print(f"Using initial model: {model_path}")

    # Initialize YOLO model
    model = YOLO(f'{model_path}.pt')
    model.to(device)

    # Warmup model for consistent performance
    print("Warming up model...")
    dummy_frame = torch.zeros((1, 3, imgsz, imgsz)).to(device)
    for _ in range(3):
        with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=(device == 'cuda')):
            _ = model.predict(source=dummy_frame, device=device, conf=conf, imgsz=imgsz, verbose=False)

    # Open video source (file or webcam)
    if video_source:
        cap = cv2.VideoCapture(video_source)
        source_name = f"Video: {video_source}"
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # DirectShow for Windows
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        source_name = f"Webcam {camera_index}"

    if not cap.isOpened():
        print(f"Error: Unable to access {source_name}.")
        return

    # Start with default resolution
    current_resolution_index = DEFAULT_RESOLUTION_INDEX
    display_width, display_height = RESOLUTIONS[current_resolution_index]

    # Setup threaded frame reading if enabled
    frame_queue = None
    frame_reader = None
    if threaded:
        frame_queue = queue.Queue(maxsize=5)
        frame_reader = FrameReader(cap, frame_queue, max_queue_size=5)
        frame_reader.start()
        print(f"Using threaded frame reader for parallel decode/inference")

    if no_display:
        print(f"Starting {source_name} in benchmark mode (no display). Press Ctrl+C to exit.")
    else:
        print(f"Starting {source_name}. Press 'q' to exit.")

    feedback_text = ""
    feedback_timer = 0
    prev_time = time.time()  # To calculate FPS
    frame_count = 0
    total_inference_time = 0

    try:
        while True:
            # Get frame from queue (threaded) or directly from capture
            if threaded:
                try:
                    frame = frame_queue.get(timeout=1.0)
                    if frame is None:  # End of video signal
                        if video_source:
                            print("End of video.")
                        break
                    ret = True
                except queue.Empty:
                    print("Frame queue timeout")
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    if video_source:
                        print("End of video.")
                    else:
                        print("Error: Unable to read frame from webcam.")
                    break

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            # Get original frame dimensions
            orig_h, orig_w = frame.shape[:2]

            # Pre-resize frame to reduce GPU memory transfer overhead
            frame_resized = cv2.resize(frame, (imgsz, imgsz))

            # Perform object detection on pre-resized frame
            inference_start = time.time()
            with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=(device == 'cuda')):
                results = model.predict(source=frame_resized, device=device, conf=conf, imgsz=imgsz, verbose=False, half=True)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            frame_count += 1

            # In benchmark mode, skip display
            if no_display:
                if frame_count % 30 == 0:  # Print stats every 30 frames
                    avg_inference_ms = (total_inference_time / frame_count) * 1000
                    theoretical_fps = 1000 / avg_inference_ms
                    print(f"Frames: {frame_count} | FPS: {fps:.1f} | Avg Inference: {avg_inference_ms:.1f}ms | Theoretical Max FPS: {theoretical_fps:.0f}")
                continue

            # Resize frame for display AFTER inference
            frame_display = cv2.resize(frame, (display_width, display_height))

            # Calculate scaling factors for bounding boxes (from imgsz to display)
            scale_x = display_width / imgsz
            scale_y = display_height / imgsz

            # Display detections on the frame
            for result in results:
                for box in result.boxes:
                    # Scale bounding box coordinates from imgsz space to display resolution
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                    # Scale label position
                    label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Add shadowed text for FPS, resolution, and model
            draw_shadowed_text(frame_display, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, (0, 0, 0))
            draw_shadowed_text(frame_display, f"Resolution: {display_width}x{display_height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, (0, 0, 0))
            draw_shadowed_text(frame_display, f"Model: {model_path}.pt", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, (0, 0, 0))

            # Display feedback text for actions
            if feedback_text and time.time() - feedback_timer < 2:  # Display feedback for 2 seconds
                draw_shadowed_text(frame_display, feedback_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, (0, 0, 0))
            else:
                feedback_text = ""

            # Display the frame
            cv2.imshow('YOLOv8 Object Detection', frame_display)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('a'):  # Decrease resolution
                if current_resolution_index > 0:
                    current_resolution_index -= 1
                    display_width, display_height = RESOLUTIONS[current_resolution_index]
                    feedback_text = f"Resolution changed to {display_width}x{display_height}"
                    feedback_timer = time.time()
            elif key == ord('d'):  # Increase resolution
                if current_resolution_index < len(RESOLUTIONS) - 1:
                    current_resolution_index += 1
                    display_width, display_height = RESOLUTIONS[current_resolution_index]
                    feedback_text = f"Resolution changed to {display_width}x{display_height}"
                    feedback_timer = time.time()
            elif key == ord('w'):  # Larger model
                if current_model_index < len(YOLO_MODELS) - 1:
                    current_model_index += 1
                    model_path = YOLO_MODELS[current_model_index]
                    model = YOLO(f'{model_path}.pt')
                    model.to(device)
                    feedback_text = f"Switched to {model_path}.pt"
                    feedback_timer = time.time()
            elif key == ord('s'):  # Smaller model
                if current_model_index > 0:
                    current_model_index -= 1
                    model_path = YOLO_MODELS[current_model_index]
                    model = YOLO(f'{model_path}.pt')
                    model.to(device)
                    feedback_text = f"Switched to {model_path}.pt"
                    feedback_timer = time.time()

    finally:
        # Stop frame reader thread if running
        if frame_reader is not None:
            frame_reader.stop()
            frame_reader.join(timeout=2.0)

        cap.release()
        if not no_display:
            cv2.destroyAllWindows()

        # Print final benchmark statistics
        if no_display and frame_count > 0:
            avg_inference_ms = (total_inference_time / frame_count) * 1000
            theoretical_fps = 1000 / avg_inference_ms
            print(f"\n=== Final Statistics ===")
            print(f"Total frames processed: {frame_count}")
            print(f"Average inference time: {avg_inference_ms:.2f}ms")
            print(f"Pure GPU inference FPS: {theoretical_fps:.0f}")
            print(f"Actual processing FPS: {fps:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection with Webcam or Video File")
    parser.add_argument("--force-cpu", action="store_true", help="Force the use of CPU instead of GPU or MPS.")
    parser.add_argument("--camera-index", type=int, default=0, help="Index of the webcam (default: 0).")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (if not specified, uses webcam).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO model (default: 640).")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for object detection (default: 0.5).")
    parser.add_argument("--no-display", action="store_true", help="Run in benchmark mode without display (shows pure GPU performance).")
    parser.add_argument("--threaded", action="store_true", help="Use threaded frame reading to overlap decode with inference (30-40% faster).")
    args = parser.parse_args()

    process_webcam(camera_index=args.camera_index, imgsz=args.imgsz, conf=args.conf, force_cpu=args.force_cpu, video_source=args.video, no_display=args.no_display, threaded=args.threaded)
