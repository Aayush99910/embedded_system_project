from typing import Any, Optional
from ultralytics import YOLO
from picamera2 import Picamera2

# importing the opencv library
import cv2

# importing the time
import time

# going to import threading in python so that we are now going to capture in one thread and then analyze in another thread
import threading

# using queue data strucure so that we can store the last image in the queue and then the analyze thread will take it and analyze
import queue

# importing os so that we can create folder for this run
import os

# importing datetime so that we can save images by date and time
from datetime import datetime

# importing air quality sensor libraries
import board
import busio
import adafruit_ccs811

# importing dht11 and buzzer libraries
import adafruit_dht
from gpiozero import OutputDevice


TARGET_CLASSES = {
    "person",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
}

ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
}

# final labels for simpler and more reliable output
DISPLAY_CLASSES = {
    "person": "person",
    "bird": "animal",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal"
}

# only accept detections if YOLO is sure enough
CONFIDENCE_LIMIT = 0.60

# imgsz controls the resolution YOLO resizes the frame to before running inference
# higher = better detection accuracy but slower, lower = faster but may miss things
# try these values and see what works best on the Pi:
#   320 -> fastest, may miss small or distant objects
#   416 -> good middle ground, a bit slower
#   480 -> better accuracy, noticeably slower
#   640 -> best accuracy, slowest — only if Pi can keep up
IMGSZ = 416

# this controls the actual camera image size that gets saved
# higher = better saved image quality but more CPU/RAM
# this is a better balance than 640x480
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

# this controls the smaller frame size used only for YOLO inference
# YOLO runs on this smaller frame so speed stays good
# the original larger frame is still saved for better image quality
YOLO_FRAME_WIDTH = 640
YOLO_FRAME_HEIGHT = 480

# jpeg quality for saved detection images
# higher = better image quality but larger file size
# 95 is near-lossless, 80 is a good balance, default opencv is around 75
JPEG_QUALITY = 90

# queue size tradeoff:
# 1 = always analyze the freshest frame, drops stale ones (best for real-time)
# 3 = small buffer so brief detections are not dropped, still close to real-time
# 4-5 = larger buffer, analyze every frame but may lag behind if YOLO is slow
QUEUE_SIZE = 3

# we will make a queue
frame_queue = queue.Queue(maxsize=QUEUE_SIZE)

# we have a flag that will start as False this is for exit condition
stop_event = threading.Event()

# making one folder for this whole run
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_FOLDER = f"detections_{RUN_TIMESTAMP}"
os.makedirs(RUN_FOLDER, exist_ok=True)

# making a log file path for this run
LOG_FILE = os.path.join(RUN_FOLDER, "run_log.txt")

# making a lock so that two threads do not write to the log file at the same time
log_lock = threading.Lock()

# making counters so that we do not print every single frame
capture_count = 0
analyze_count = 0

# making one more lock for updating counters safely
counter_lock = threading.Lock()

# storing the last valid air sensor readings
last_eco2 = 400
last_tvoc = 0
air_lock = threading.Lock()

# storing the last valid dht11 readings
last_temp_f = 72.0
last_humidity = 0
dht_lock = threading.Lock()

# lock for buzzer so threads do not conflict
buzzer_lock = threading.Lock()


# helper function so that we can both print and save logs into file
def log_message(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"

    with log_lock:
        print(full_message)
        with open(LOG_FILE, "a") as f:
            f.write(full_message + "\n")


# helper function to read air quality sensor values
def read_air_sensor(ccs):
    global last_eco2, last_tvoc

    try:
        if ccs.data_ready:
            eco2 = ccs.eco2
            tvoc = ccs.tvoc

            with air_lock:
                last_eco2 = eco2
                last_tvoc = tvoc

            return eco2, tvoc
        else:
            with air_lock:
                return last_eco2, last_tvoc
    except Exception:
        with air_lock:
            return last_eco2, last_tvoc


# helper function to read dht11 values
def read_dht11(dht_device):
    global last_temp_f, last_humidity

    try:
        temperature_c = dht_device.temperature
        humidity = dht_device.humidity

        if temperature_c is not None and humidity is not None:
            temperature_f = (temperature_c * 9 / 5) + 32

            with dht_lock:
                last_temp_f = temperature_f
                last_humidity = humidity

            return temperature_f, humidity
        else:
            with dht_lock:
                return last_temp_f, last_humidity

    except RuntimeError:
        with dht_lock:
            return last_temp_f, last_humidity
    except Exception:
        with dht_lock:
            return last_temp_f, last_humidity


# helper function for buzzer alert
def person_alert(buzzer):
    with buzzer_lock:
        for _ in range(2):
            buzzer.on()
            time.sleep(0.12)
            buzzer.off()
            time.sleep(0.12)


# First I will be taking a picture from the picamera
def capture_a_frame(picam2) -> Optional[Any]:
    frame = picam2.capture_array()

    # if we have the picture we will return the frame
    if frame is not None:
        return frame
    else:
        log_message("Camera not working!")
        return None


# now a function to analyze the picture that was taken
def analyze_frame(frame, model, ccs, dht_device, buzzer) -> None:
    inference_start = time.time()

    # keeping the original frame high quality so saved images look better
    original_frame = frame.copy()

    # resizing only for YOLO so inference stays fast
    yolo_frame = cv2.resize(frame, (YOLO_FRAME_WIDTH, YOLO_FRAME_HEIGHT))

    # using the IMGSZ constant defined at the top — change that value to tune accuracy vs speed
    results = model(yolo_frame, imgsz=IMGSZ, verbose=False)

    inference_end = time.time()

    # calculate the scale so boxes from the smaller YOLO frame match the larger saved image
    scale_x = original_frame.shape[1] / YOLO_FRAME_WIDTH
    scale_y = original_frame.shape[0] / YOLO_FRAME_HEIGHT

    # flag to see if anything is detected or not
    detected = False
    person_detected = False
    animal_detected = False

    # storing what classes were detected
    detected_classes = []

    draw_start = time.time()

    # results give you all the details of the result
    for result in results:
        # boxes are if the model detects anything that the model is trained on
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)

            # only allow target animals/people AND only if confidence is high enough
            if class_name in DISPLAY_CLASSES and confidence >= CONFIDENCE_LIMIT:
                display_name = DISPLAY_CLASSES[class_name]
                detected = True
                detected_classes.append(f"{display_name} {round(confidence * 100)}%")

                if display_name == "person":
                    person_detected = True
                elif display_name == "animal":
                    animal_detected = True

                # get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # scale box coordinates back to the original high quality frame
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                # draw the box
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # draw the label
                label = f"{display_name} {round(confidence * 100)}%"

                # write that text into the frame
                cv2.putText(
                    original_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    draw_end = time.time()

    # reading air sensor values
    eco2, tvoc = read_air_sensor(ccs)

    # reading dht11 values
    temperature_f, humidity = read_dht11(dht_device)

    # getting current date and time
    current_time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # making text for sensors
    air_text_1 = f"eCO2: {eco2} ppm"
    air_text_2 = f"TVOC: {tvoc} ppb"
    dht_text_1 = f"Temp: {temperature_f:.1f} F"
    dht_text_2 = f"Humidity: {humidity} %"

    # writing air quality values on the image - top left
    cv2.putText(
        original_frame,
        air_text_1,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        original_frame,
        air_text_2,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # writing date and time at the bottom left
    cv2.putText(
        original_frame,
        current_time_text,
        (10, original_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    # writing temperature and humidity at the extreme bottom right
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 5

    frame_height, frame_width = original_frame.shape[:2]

    (temp_width, temp_height), _ = cv2.getTextSize(dht_text_1, font, font_scale, thickness)
    (hum_width, hum_height), _ = cv2.getTextSize(dht_text_2, font, font_scale, thickness)

    temp_x = frame_width - temp_width - margin
    temp_y = frame_height - hum_height - 20

    hum_x = frame_width - hum_width - margin
    hum_y = frame_height - margin

    cv2.putText(
        original_frame,
        dht_text_1,
        (temp_x, temp_y),
        font,
        font_scale,
        (255, 200, 0),
        thickness
    )

    cv2.putText(
        original_frame,
        dht_text_2,
        (hum_x, hum_y),
        font,
        font_scale,
        (255, 200, 0),
        thickness
    )

    # IMAGE MODE: no cv2.imshow() and no cv2.waitKey() here
    # we only save a JPEG when something is actually detected
    # to stop the program press Ctrl+C in the terminal

    # buzzer alert only if person is detected
    if person_detected:
        person_alert(buzzer)

    if detected:
        save_start = time.time()

        filename = f"detection_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        filepath = os.path.join(RUN_FOLDER, filename)

        # using JPEG_QUALITY constant so saved images are not blurry or compressed
        # change JPEG_QUALITY at the top of the file to tune file size vs image quality
        cv2.imwrite(filepath, original_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        save_end = time.time()

        log_message(f"DETECTED: {', '.join(detected_classes)}")
        log_message(f"Person detected: {person_detected}")
        log_message(f"Animal detected: {animal_detected}")
        log_message(f"Air Quality -> eCO2: {eco2} ppm, TVOC: {tvoc} ppb")
        log_message(f"DHT11 -> Temperature: {temperature_f:.1f} F, Humidity: {humidity} %")
        log_message(f"YOLO inference time: {inference_end - inference_start:.4f} seconds")
        log_message(f"Drawing boxes and labels time: {draw_end - draw_start:.4f} seconds")
        log_message(f"Image save time: {save_end - save_start:.4f} seconds")
        log_message(f"Saved to {filepath}")


# writing the thread for capture
def capture_worker(picam2) -> None:
    global capture_count

    # running infintely until stopped
    while not stop_event.is_set():
        capture_start = time.time()
        frame = capture_a_frame(picam2)
        capture_end = time.time()

        # this is if the queue is full then we will drain it and replace with the new image that we captured
        # draining instead of removing just one makes sure the newest frame always gets in
        if frame is not None:
            if frame_queue.full():
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        break

            # we will put the new picture
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        with counter_lock:
            capture_count += 1
            current_capture_count = capture_count

        if current_capture_count % 30 == 0:
            log_message(f"Capture time: {capture_end - capture_start:.4f} seconds")


# now writing the analyzer model
def analyze_worker(model, ccs, dht_device, buzzer) -> None:
    global analyze_count

    # doing this continously until stopped as well
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        analyze_time_start = time.time()
        analyze_frame(frame, model, ccs, dht_device, buzzer)
        analyze_time_end = time.time()

        with counter_lock:
            analyze_count += 1
            current_analyze_count = analyze_count

        if current_analyze_count % 30 == 0:
            log_message(f"Analyze time (YOLO + draw + save): {analyze_time_end - analyze_time_start:.4f} seconds")


def main() -> None:
    start = time.time()

    log_message("Program started (IMAGE mode — no live window, saves JPEGs on detection only)")
    log_message(f"Run folder created: {RUN_FOLDER}")
    log_message(f"YOLO imgsz set to: {IMGSZ} — change IMGSZ at the top of the file to tune accuracy vs speed")
    log_message(f"Camera image size set to: {CAMERA_WIDTH}x{CAMERA_HEIGHT} — change CAMERA_WIDTH and CAMERA_HEIGHT at the top of the file to tune saved image quality vs speed")
    log_message(f"YOLO frame size set to: {YOLO_FRAME_WIDTH}x{YOLO_FRAME_HEIGHT} — change YOLO_FRAME_WIDTH and YOLO_FRAME_HEIGHT at the top of the file to tune inference speed")
    log_message(f"JPEG quality set to: {JPEG_QUALITY} — change JPEG_QUALITY at the top of the file to tune image quality vs file size")

    model_load_start = time.time()

    # yolov8n is lighter and faster on Raspberry Pi
    model = YOLO("yolov8n.pt")

    model_load_end = time.time()
    log_message(f"Model load time: {model_load_end - model_load_start:.4f} seconds")

    # setting up the air quality sensor
    i2c = busio.I2C(board.SCL, board.SDA)
    ccs = adafruit_ccs811.CCS811(i2c)

    log_message("Waiting for air sensor to be ready...")
    while not ccs.data_ready:
        time.sleep(1)
    log_message("Air sensor is ready")

    # get one valid reading before starting live view
    eco2, tvoc = read_air_sensor(ccs)
    log_message(f"Initial Air Quality -> eCO2: {eco2} ppm, TVOC: {tvoc} ppb")

    # setting up dht11
    dht_device = adafruit_dht.DHT11(board.D27)
    temperature_f, humidity = read_dht11(dht_device)
    log_message(f"Initial DHT11 -> Temperature: {temperature_f:.1f} F, Humidity: {humidity} %")

    # setting up buzzer
    buzzer = OutputDevice(17, active_high=False, initial_value=False)

    camera_init_start = time.time()
    picam2 = Picamera2()
    camera_init_end = time.time()
    log_message(f"Camera object creation time: {camera_init_end - camera_init_start:.4f} seconds")

    config_start = time.time()

    # setting up the camera configuration
    config = picam2.create_video_configuration(
        raw={"size": (1640, 1232)},
        main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
        buffer_count=1,
        queue=False
    )

    picam2.configure(config)

    config_end = time.time()
    log_message(f"Camera configuration time: {config_end - config_start:.4f} seconds")

    camera_start_start = time.time()
    picam2.start()
    camera_start_end = time.time()
    log_message(f"Camera start time: {camera_start_end - camera_start_start:.4f} seconds")

    log_message("Running headlessly — press Ctrl+C to stop")

    capture_thread = threading.Thread(target=capture_worker, args=(picam2,), daemon=True)
    analyze_thread = threading.Thread(
        target=analyze_worker,
        args=(model, ccs, dht_device, buzzer),
        daemon=True
    )

    # we will now create these two threads
    capture_thread.start()
    analyze_thread.start()

    # now running them simultaneously
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log_message("Stopping Threads...")
        stop_event.set()

    # wait for both threads to fully finish before continuing
    capture_thread.join()
    analyze_thread.join()

    picam2.stop()
    cv2.destroyAllWindows()
    buzzer.off()
    dht_device.exit()

    end = time.time()

    # total time it took
    log_message(f"Total time that it took: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()