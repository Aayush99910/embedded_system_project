from typing import Any, Optional
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import time

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

# capture a frame from the pi camera
def capture_a_frame(picam2) -> Optional[Any]:
    frame = picam2.capture_array()
    
    # RGB888 format gives 3 channels, just swap R and B for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    if frame is not None:
        return frame
    else:
        print("Camera not working!")
        return None

# analyze the frame with YOLO and draw bounding boxes
def analyze_frame(frame, model) -> None:
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            # get class name and confidence
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)

            if class_name in TARGET_CLASSES:
                # get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # draw label
                label = f"{class_name} {round(confidence * 100)}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"DETECTED: {class_name} — {round(confidence * 100)}% confident")

    # show the frame
    cv2.imshow("Backyard Monitor", frame)

def main() -> None:
    model = YOLO("yolov8n.pt")
    picam2 = Picamera2()

    # configure camera with correct format and resolution
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (2560, 1440)})
    picam2.configure(config)
    picam2.start()

    # warmup
    time.sleep(2)

    print("Press Q to quit")

    while True:
        frame = capture_a_frame(picam2)
        if frame is None:
            break

        analyze_frame(frame, model)

        # 1ms delay and check for Q key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
