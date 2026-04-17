#  Smart Backyard Monitor

A real-time backyard security and environmental monitoring system built on a **Raspberry Pi 5**. A USB webcam captures one frame per second, runs it through **YOLOv8 nano** to detect people and animals, and logs every detection with a timestamp. GPIO sensors for temperature, humidity, and air quality are handled in **C**.

Built as an embedded systems class project.

**Team:** Aayush Maharjan + Husam Sankar

---

## What It Does

- Captures one frame per second from a USB webcam
- Runs YOLOv8 object detection on every frame
- Detects: **people, cats, dogs, birds, squirrels** (and more)
- Logs temperature + humidity from a DHT22 sensor (C)
- Reads air quality from an MQ-135 sensor via SPI (C)
- PIR motion sensor triggers camera capture (interrupt-driven, C)
- Logs all detections and sensor readings with timestamps

---

## Hardware

| Part | Purpose |
|------|---------|
| Raspberry Pi 5 (2GB) | Main compute board |
| Aluratek HD Webcam (AWC04F) | Camera input — USB plug and play |
| HC-SR501 PIR Motion Sensor | Interrupt-driven motion trigger |
| DHT22 Sensor | Temperature + humidity readings |
| MQ-135 Air Quality Sensor | Analog air quality (via MCP3008 ADC) |
| MCP3008 ADC Chip | Reads analog MQ-135 output via SPI |
| 2x resistors (voltage divider) | Steps MQ-135 5V output down to 3.3V |
| 10kΩ resistors (x3) | Pull-ups for DHT22 + PIR |

> **MQ-135 Warning:** Outputs 5V — always use a voltage divider before the MCP3008. Running MCP3008 at 5V will damage the Pi.

> **DHT22 Note:** Use Adafruit CircuitPython DHT library only. pigpio is NOT supported on Pi 5.

---

## Tech Stack

| Layer | Language / Library | Purpose |
|-------|--------------------|---------|
| Camera capture | Python + OpenCV | Grab frame from USB webcam every 1 second |
| Object detection | Python + Ultralytics YOLOv8 | Detect people and animals in each frame |
| GPIO & sensors | C (bcm2835 / wiringPi) | PIR interrupts, DHT22, MQ-135 via SPI |
| Output / logging | Python | Print detections + log to CSV |

---

## Project Structure

```
embedded_system_project/
│
├── python/
│   └── detect.py          # Main YOLO detection script
│
├── c/
│   ├── pir_sensor.c        # PIR motion interrupt handler
│   ├── dht22_reader.c      # Temperature + humidity logger
│   └── mq135_reader.c      # Air quality via SPI + MCP3008
│
├── logs/
│   ├── detections.csv      # YOLO detection log
│   └── sensors.csv         # Temp, humidity, air quality log
│
└── README.md
```

---

## Setup

### 1. Flash the Pi

Flash **Raspberry Pi OS 64-bit (Bookworm)** to a 64GB SD card using Raspberry Pi Imager.

### 2. Install Python Dependencies

```bash
pip install ultralytics opencv-python
```

YOLOv8 nano model weights (~6MB) download automatically on first run.

### 3. Install C Library

```bash
# bcm2835
wget http://www.airspayce.com/mikem/bcm2835/bcm2835-1.71.tar.gz
tar zxvf bcm2835-1.71.tar.gz
cd bcm2835-1.71 && ./configure && make && sudo make install
```

### 4. Connect the Webcam

Plug the USB webcam into any USB port on the Pi. Verify it shows up:

```bash
ls /dev/video*
# Should output: /dev/video0
```

### 5. Run Detection

```bash
python python/detect.py
```

---

## How It Works

### Detection Pipeline

```
[USB Webcam] → capture frame every 1s → [YOLOv8 nano] → print detections
```

### PIR Integration (Phase 2)

```
[PIR Sensor] → [C daemon] → trigger signal → [Python YOLO script] → save snapshot
```

### Performance

| Task | Time |
|------|------|
| Frame capture | ~5–10ms |
| YOLOv8 nano inference (Pi 5) | ~200–400ms |
| Total per loop | ~210–410ms |
| Budget (1fps = 1000ms) | 590ms to spare |

---

## Roadmap

- [x] USB webcam capture at 1fps
- [x] YOLOv8 nano object detection
- [x] Filter detections to people + animals
- [ ] PIR motion sensor interrupt in C
- [ ] DHT22 temperature + humidity logging
- [ ] MQ-135 air quality via SPI
- [ ] PIR triggers YOLO pipeline
- [ ] Save annotated snapshots with bounding boxes
- [ ] Telegram phone alerts on detection
- [ ] Flask web dashboard

---

## Wiring Notes

**DHT22**
- VCC → 3.3V
- GND → GND
- DATA → GPIO pin + 10kΩ pull-up resistor to 3.3V

**PIR (HC-SR501)**
- VCC → 5V
- GND → GND
- DATA → GPIO pin (output is 3.3V safe)

**MQ-135 + MCP3008**
- MQ-135 VCC → 5V, output → voltage divider → MCP3008 CH0
- MCP3008 VDD + VREF → 3.3V (do NOT use 5V)
- MCP3008 → Pi via SPI (CLK, MOSI, MISO, CS)

---

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [EJ Technology — YOLO on Raspberry Pi](https://ejtech.io)
- [Random Nerd Tutorials — DHT22 on Pi](https://randomnerdtutorials.com)
- [bcm2835 C Library](http://www.airspayce.com/mikem/bcm2835/)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
