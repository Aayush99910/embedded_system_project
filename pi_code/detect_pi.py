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

<<<<<<< HEAD
# importing os so that we can create folder for this run
import os

# importing datetime so that we can save images by date and time
from datetime import datetime

=======
>>>>>>> 125869e (doing final tests and works)
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

# we will make a queue
frame_queue = queue.Queue(maxsize=1)

# we have a flag that will start as False this is for exit condition
stop_event = threading.Event()

<<<<<<< HEAD
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


# helper function so that we can both print and save logs into file
def log_message(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"

    with log_lock:
        print(full_message)
        with open(LOG_FILE, "a") as f:
            f.write(full_message + "\n")

=======
>>>>>>> 125869e (doing final tests and works)

# First I will be taking a picture from the picamera
def capture_a_frame(picam2) -> Optional[Any]: 
    frame = picam2.capture_array() 
   
    # resizing down for faster tlo inference but keep wide FOV
    # frame = cv2.resize(frame, (640, 480))

    # convert the RGBA to RBG (PI camera gives 4 channels) 
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # if we have the picture we will return the frame
    if frame is not None: 
        return frame 
    else:
        log_message("Camera not working!")
        return None
    

# now a function to analyze the picture that was taken
def analyze_frame(frame, model) -> None:
    inference_start = time.time()
<<<<<<< HEAD
    results = model(frame, imgsz=320, verbose=False)
=======
    results = model(frame,imgsz=320, verbose=False)
>>>>>>> 125869e (doing final tests and works)
    inference_end = time.time()

    # flag to see if anything is detected or not 
    detected = False 

    # storing what classes were detected
    detected_classes = []

    # flag to see if anything is detected or not 
    detected = False 

    draw_start = time.time()

    # results give you all the details of the result we can see it in the CLI I have printed it above
    for result in results:
        # boxes are if the model detects anything that the model is trained on
        # person, animal, any object
        for box in result.boxes:
            # box.cls gives you a number 0, 1, or anything 
            # then model.names is the dictionary of key -> value 
            # one example 0 -> "person"
            # so class_name gives you person, cow, etc 
            # box.conf gives you the percentage of confidence
            # then confidence is just converting that box.conf into floating number type
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            
            # if we are having the class_name in the target_class above then we go inside the if here
            if class_name in TARGET_CLASSES:
                # setting the flag to be True 
                detected = True 
<<<<<<< HEAD
                detected_classes.append(f"{class_name} {round(confidence * 100)}%")
=======
>>>>>>> 125869e (doing final tests and works)

                # get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # box.xyxy gives me the four corner of the frame and then map just converts that to integer number
                # x1 y1 are top left corner and then # x2 y2 are bottom right
                
                # draw the box
                # we have the frame the window where this will write this thing on
                # we have the two points 
                # cv2 figures out the other two points by itselfs
                # then I have color and thickness
                # (frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # draw the label
                # f string to print everything nicely
                label = f"{class_name} {round(confidence * 100)}%"

                # write that text into the frame
                # (frame, label, (x1, y1 - 10), cv2. FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # print statement
                # print(f"DETECTED: {class_name} — {round(confidence * 100)}% confident")
    
    draw_end = time.time()

    # show the frame with boxes drawn on it
    #cv2.imshow("Backyard Monitor", frame)
    #cv2.waitKey(0)
    
    if detected:
        save_start = time.time()
<<<<<<< HEAD
        filename = f"detection_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        filepath = os.path.join(RUN_FOLDER, filename)
        #cv2.imwrite("lastcapture.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        cv2.imwrite(filepath, frame)
        save_end = time.time()

        log_message(f"DETECTED: {', '.join(detected_classes)}")
        log_message(f"YOLO inference time: {inference_end - inference_start:.4f} seconds")
        log_message(f"Drawing boxes and labels time: {draw_end - draw_start:.4f} seconds")
        log_message(f"Image save time: {save_end - save_start:.4f} seconds")
        log_message(f"Saved to {filepath}")
=======
        filename = f"lastcapture_{int(time.time())}.jpg"
        #cv2.imwrite("lastcapture.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        cv2.imwrite(filename, frame)
        save_end = time.time()
        print(f"Image save time: {save_end - save_start:.4f} seconds")
        print(f"Saved to {filename}")
>>>>>>> 125869e (doing final tests and works)


# writing the thread for capture 
def capture_worker(picam2) -> None:
<<<<<<< HEAD
    global capture_count

=======
>>>>>>> 125869e (doing final tests and works)
    # running infintely until stopped
    while not stop_event.is_set():
        capture_start = time.time()
        frame = capture_a_frame(picam2)
        capture_end = time.time() 
        
        # this is if the queue is full then we will replace that with the new image that we captured 
        # we will pass if its empty
        # and we will put the new frame in the queue
        if frame is not None: 
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass 
<<<<<<< HEAD

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
=======
            else:
                # we will put the new picture 
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass 
        print(f"Capture time: {capture_end - capture_start:.4f} seconds")
>>>>>>> 125869e (doing final tests and works)
        

# now writing the analyzer model
def analyze_worker(model) -> None: 
<<<<<<< HEAD
    global analyze_count

=======
>>>>>>> 125869e (doing final tests and works)
    # doing this continously until stopped as well
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue 
    
        analyze_time_start = time.time()
        analyze_frame(frame, model)
        analyze_time_end = time.time()
<<<<<<< HEAD

        with counter_lock:
            analyze_count += 1
            current_analyze_count = analyze_count
    
        if current_analyze_count % 30 == 0:
            log_message(f"Analyze time (YOLO + draw + save): {analyze_time_end - analyze_time_start:.4f} seconds")
=======
    
    print(f"Analyze time (YOLO + draw + save): {analyze_time_end - analyze_time_start:.4f} seconds")
>>>>>>> 125869e (doing final tests and works)
    
                


def main() -> None:
    start = time.time()

    log_message("Program started")
    log_message(f"Run folder created: {RUN_FOLDER}")

    model_load_start = time.time()
    model = YOLO("yolov8n.pt") # we are using the yolo v8 nano because they are very lightweight and also fast
    model_load_end = time.time()
    log_message(f"Model load time: {model_load_end - model_load_start:.4f} seconds")

    camera_init_start = time.time()
    picam2 = Picamera2()# we are connecting to the camera 
    camera_init_end = time.time()
    log_message(f"Camera object creation time: {camera_init_end - camera_init_start:.4f} seconds")

    config_start = time.time()
    # we are setting up the camera configuration.
    #config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    config = picam2.create_video_configuration(raw={"size": (1640, 1232)}, main={"format": "RGB888", "size": (640, 480)}, buffer_count=1, queue=False)
    picam2.configure(config)
    config_end = time.time()
    log_message(f"Camera configuration time: {config_end - config_start:.4f} seconds")

    camera_start_start = time.time()
    picam2.start() # and here we are turning the camera on and starting to capture the image
    camera_start_end = time.time()
    log_message(f"Camera start time: {camera_start_end - camera_start_start:.4f} seconds")

    capture_thread = threading.Thread(target=capture_worker, args=(picam2,), daemon=True)
    analyze_thread = threading.Thread(target=analyze_worker, args=(model,), daemon=True)
    
    # we will now create these two threads
    capture_thread.start()
    analyze_thread.start()
    
    # now running them simultaneously
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
<<<<<<< HEAD
        log_message("Stopping Threads...")
=======
        print("Stopping Threads...")
>>>>>>> 125869e (doing final tests and works)
        stop_event.set()
    
    # wait for both threads to fully finish before continuing 
    capture_thread.join()
    analyze_thread.join()
    
    picam2.stop()

    end = time.time()

    # total time it took 
    log_message(f"Total time that it took: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
