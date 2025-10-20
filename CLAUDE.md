# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time object detection application using YOLO11/YOLOv8 and PyTorch for webcams and video files. Optimized for GPU acceleration (NVIDIA CUDA) with automatic CPU fallback. Defaults to YOLO11 models for best performance.

## Development Environment

**Python Version**: Managed via `.python-version` file

**GPU Support**: PyTorch with CUDA 11.8 (`torch==2.5.1+cu118`)

**Key Dependencies**:
- `ultralytics==8.3.217` - YOLO11 and YOLOv8 implementation
- `opencv-python==4.10.0.84` - Video capture and display
- `torch==2.5.1+cu118` - GPU acceleration
- `PyQt5` - GUI components (used for potential future features)

**Install Dependencies**:
```bash
pip install -r requirements.txt
```

## Running the Application

**Basic webcam detection**:
```bash
python object_detection.py
```

**Video file processing**:
```bash
python object_detection.py --video path/to/video.mp4
```

**Benchmark mode (no display, pure GPU performance)**:
```bash
python object_detection.py --video path/to/video.mp4 --no-display
```

**Threaded mode (30-40% faster)**:
```bash
python object_detection.py --threaded
```

**Common options**:
- `--camera-index 0` - Select webcam (default: 0)
- `--imgsz 640` - Model input resolution for inference (default: 640)
- `--conf 0.5` - Confidence threshold (default: 0.5)
- `--force-cpu` - Force CPU mode (disable GPU acceleration)

**Device Selection (CPU vs CUDA)**:
- Device is selected **at startup** and cannot be changed during runtime
- Auto-detects CUDA GPU availability via `torch.cuda.is_available()`
- Use `--force-cpu` flag to override GPU detection and force CPU mode
- Current device is displayed at startup (e.g., "Using device: NVIDIA GeForce RTX 3090")

## Interactive Runtime Controls

When the application is running with display enabled, use these keypresses to adjust settings in real-time:

**Model Switching** (lines 235-250):
- `W` - Switch to larger/more accurate model
  - Cycles up through: n → s → m → l → x
  - Higher models = better accuracy, lower FPS
  - Model loads immediately and displays feedback message
- `S` - Switch to smaller/faster model
  - Cycles down through: x → l → m → s → n
  - Smaller models = higher FPS, lower accuracy
  - Current model displayed in overlay (e.g., "Model: yolov8x.pt")

**Display Resolution Adjustment** (lines 223-234):
- `A` - Decrease display resolution
  - Cycles down through: 1080p → 720p → 480p
  - Does NOT affect inference resolution (always uses `--imgsz`)
  - Reduces rendering overhead, may improve FPS
- `D` - Increase display resolution
  - Cycles up through: 480p → 720p → 1080p
  - Higher resolution = better visual quality
  - Available resolutions: 854x480, 1280x720, 1920x1080 (16:9 aspect ratio)
  - Current resolution displayed in overlay (e.g., "Resolution: 1920x1080")

**General Controls**:
- `Q` - Quit application and display final statistics

**Visual Feedback**:
- All setting changes show a temporary feedback message for 2 seconds
- FPS counter updates in real-time to reflect performance changes
- Model and resolution info always visible in top-left overlay

## Available YOLO Models

**YOLO11 Models** (current, recommended):
- `yolo11n.pt` - Nano (fastest, least accurate)
- `yolo11s.pt` - Small (default on CPU)
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra large (default on GPU, most accurate)

**YOLOv8 Models** (legacy, still supported):
- `yolov8n.pt` - Nano
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra large

**Default selection**: `yolo11x` on GPU, `yolo11s` on CPU

**Performance**: YOLO11 models are ~10-15% faster and ~5-10% more accurate than equivalent YOLOv8 models.

**Note**: All models auto-download from Ultralytics on first use (not included in repository).

## Benchmarking

**Pure GPU inference benchmark**:
```bash
python benchmark_pure_gpu.py
```

Runs 500 frames through YOLOv8x model and reports:
- Average inference time (ms)
- Pure GPU FPS (theoretical maximum)

## Building Standalone Executable

**PyInstaller spec file**: `object_detection.spec`

```bash
pyinstaller object_detection.spec
```

Output: `dist/object_detection.exe` (Windows)

## Architecture

### Main Script: `object_detection.py`

**Core Components**:

1. **Model Selection** (lines 20-26): Dual model support
   - YOLO11 models (default, lines 24): Current generation, better performance
   - YOLOv8 models (legacy, line 22): Backward compatibility
   - Runtime switching via W/S keys allows toggling between all 10 models

2. **FrameReader Thread** (lines 32-58): Asynchronous frame capture
   - Decouples video decode (CPU) from inference (GPU)
   - Uses queue to buffer frames (max 5)
   - Enables 30-40% performance boost with `--threaded` flag

3. **DPI Awareness** (lines 14-18): Windows-specific fix
   - Prevents 200% scaling issues on high-DPI displays
   - Uses `ctypes.windll.shcore.SetProcessDpiAwareness(1)`

4. **Model Warmup** (lines 99-105): Performance optimization
   - Runs 3 dummy inferences before processing
   - Ensures consistent timing measurements
   - Uses `torch.amp.autocast` for mixed precision

5. **Two-Stage Rendering Pipeline**:
   - **Inference Stage**: Frame resized to `imgsz` (default 640x640)
   - **Display Stage**: Frame scaled to selected display resolution (480p/720p/1080p)
   - Bounding boxes scaled from inference resolution to display resolution

6. **Shadowed Text Rendering** (lines 57-75):
   - Translucent background boxes for readability
   - Used for FPS, resolution, and model info overlays
   - Uses `cv2.addWeighted` for alpha blending

**Performance Optimizations**:
- Mixed precision inference (`torch.amp.autocast` + `half=True`)
- Minimal buffer lag (`cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`)
- DirectShow API on Windows (`cv2.CAP_DSHOW`)
- Pre-resize frames before inference to reduce GPU transfer overhead

### Benchmark Script: `benchmark_pure_gpu.py`

Pure GPU inference timing without video I/O or display overhead. Uses random numpy frame to isolate model performance.

## Platform-Specific Notes

**Windows**:
- Uses DirectShow (`cv2.CAP_DSHOW`) for webcam access
- DPI awareness required for proper scaling
- PyInstaller builds supported

**GPU Requirements**:
- NVIDIA GPU with CUDA 11.8 support
- Automatic fallback to CPU if CUDA unavailable
- Device info displayed at startup

## Video Files in Repository

Contains sample video files for testing:
- `2bg-24.mkv` (1.6GB)
- `2bg-60.mp4` (46MB)
- `31422484-a336-4a49-9f61-4c074510d2d2.mp4` (1GB)

These are test assets, not part of the core codebase.

## Code Style Notes

- Uses `torch.amp.autocast` (not deprecated `torch.cuda.amp.autocast`)
- Threading implemented via `threading.Thread` with daemon threads
- Error handling for webcam access and frame reading
- CLI uses `argparse` for all configuration
