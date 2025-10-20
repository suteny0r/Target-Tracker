# Target Tracker

Real-time object detection using YOLO11/YOLOv8 for webcams and video files. Optimized for NVIDIA GPUs with CUDA support and automatic CPU fallback.

![Version](https://img.shields.io/badge/version-1.1.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![CUDA](https://img.shields.io/badge/CUDA-11.8-orange)

## Features

- **YOLO11 Support**: Latest YOLO11 models with 10-15% better performance than YOLOv8
- **GPU Accelerated**: Optimized for NVIDIA GPUs with CUDA 11.8
- **Real-time Detection**: Webcam and video file support
- **Interactive Controls**: Switch models and resolutions during runtime
- **Threaded Processing**: 30-40% performance boost with parallel frame reading
- **10 Models Available**: 5 YOLO11 models + 5 YOLOv8 models (backward compatible)

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 11.8 (optional, auto-falls back to CPU)
- Webcam (for live detection)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/suteny0r/Target-Tracker.git
cd Target-Tracker
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install PyTorch with CUDA** (for GPU acceleration)
```bash
pip install torch==2.5.1+cu118 torchaudio==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:
```bash
pip install torch torchvision torchaudio
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

**Basic webcam detection:**
```bash
python object_detection.py
```

**Process video file:**
```bash
python object_detection.py --video path/to/video.mp4
```

**Threaded mode (30-40% faster):**
```bash
python object_detection.py --threaded
```

**GPU Benchmark mode:**
```bash
python object_detection.py --no-display --threaded
```

### Interactive Controls

While running, use these keys:

- `W` / `S` - Switch to larger/smaller model (10 models available)
- `A` / `D` - Decrease/increase display resolution (480p/720p/1080p)
- `Q` - Quit

## Available Models

### YOLO11 (Current, Recommended)
- `yolo11n` - Nano (fastest, ~100+ FPS on RTX 3090)
- `yolo11s` - Small (default on CPU)
- `yolo11m` - Medium
- `yolo11l` - Large
- `yolo11x` - Extra Large (default on GPU, most accurate)

### YOLOv8 (Legacy, Still Supported)
- `yolov8n` through `yolov8x` (same size variants)

**Performance**: YOLO11 models are ~10-15% faster and ~5-10% more accurate than YOLOv8.

**Note**: Models auto-download from Ultralytics on first use (~6MB to 131MB each).

## Command-Line Options

```
--camera-index 0        Select webcam (default: 0)
--video PATH           Path to video file
--imgsz 640            Model input resolution (default: 640)
--conf 0.5             Confidence threshold (default: 0.5)
--threaded             Enable threaded frame reading (30-40% faster)
--no-display           Benchmark mode without display
--force-cpu            Force CPU mode (disable GPU)
```

## Performance

On NVIDIA RTX 3090 with YOLO11x:
- **Threaded mode**: ~60-80 FPS (1080p display)
- **Benchmark mode**: ~100+ FPS (no display overhead)
- **CPU mode**: ~10-15 FPS

## Building Standalone Executable

```bash
pyinstaller object_detection.spec
```

Output: `dist/object_detection.exe`

## Project Structure

```
Target-Tracker/
├── object_detection.py      # Main application
├── benchmark_pure_gpu.py   # GPU benchmark script
├── requirements.txt        # Python dependencies
├── CLAUDE.md              # Developer documentation
├── README.md              # This file
└── venv/                  # Virtual environment (not in git)
```

## Troubleshooting

**CUDA not detected:**
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Check CUDA 11.8 compatibility with your GPU
- Fallback to CPU mode with `--force-cpu`

**Low FPS:**
- Use `--threaded` flag
- Switch to smaller model (press `S`)
- Reduce display resolution (press `A`)
- Lower `--imgsz` value

**Models not downloading:**
- Check internet connection
- Ultralytics will auto-download on first use
- Manual download: https://docs.ultralytics.com/

## Version History

- **v1.1.0** (2025-10-20): YOLO11 upgrade, 10 models available, performance improvements
- **v1.0.0** (2025-10-20): Initial release with YOLOv8

## License

See repository for license details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO11 and YOLOv8 implementation
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
