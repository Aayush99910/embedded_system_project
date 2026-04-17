# 🎥 Smart Backyard Monitor

A real-time backyard animal and person detector built on a **Raspberry Pi 5**. A USB webcam captures one frame per second and runs it through **YOLOv8 nano** to detect people and animals. That's it.

Built as an embedded systems class project.

**Team:** Aayush Maharjan + Husam Sankar

---

## What It Does

-  **Input:** USB webcam captures one frame per second
-  **Processing:** YOLOv8 nano runs object detection on each frame
-  **Output:** Prints what was detected — people, cats, dogs, birds, squirrels

---

## Hardware

| Part | Purpose |
|------|---------|
| Raspberry Pi 5 (2GB) | Main compute board |
| Aluratek HD Webcam (AWC04F) | Camera input — USB plug and play |

---

## Tech Stack

| Layer | Library | Purpose |
|-------|---------|---------|
| Camera capture | Python + OpenCV | Grab one frame per second from USB webcam |
| Object detection | Python + Ultralytics YOLOv8 nano | Detect people and animals in each frame |

---

## Project Structure

```
embedded_system_project/
│
├── detect.py       # Main script — captures frame and runs YOLO
└── README.md
```

---

## Setup

### 1. Flash the Pi

Flash **Raspberry Pi OS 64-bit (Bookworm)** to a 64GB SD card using Raspberry Pi Imager.

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python
```

YOLOv8 nano model weights (~6MB) download automatically on first run.

### 3. Connect the Webcam

Plug the USB webcam into any USB port on the Pi. Verify it shows up:

```bash
ls /dev/video*
# Should output: /dev/video0
```

### 4. Run

```bash
python detect.py
```

---

## How It Works

```
[USB Webcam] → capture 1 frame per second → [YOLOv8 nano] → print what was detected
```

### Detection Classes

People and animals only:

- person
- cat
- dog
- bird
- squirrel

### Performance

| Task | Time |
|------|------|
| Frame capture | ~5–10ms |
| YOLOv8 nano inference on Pi 5 | ~200–400ms |
| Total per loop | ~210–410ms |
| Budget (1fps = 1000ms) |  Comfortably within limit |

---

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [EJ Technology — YOLO on Raspberry Pi](https://ejtech.io)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
