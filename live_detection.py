from typing import Any, Optional
from ultralytics import YOLO
import cv2 

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

for i in range(5):
    cap.read()

def capture_a_frame(cap) -> Optional[Any]: 
    ret, frame = cap.read()
    if ret: 
        return frame 
    else:
        print("Camera not working!")
        return None

def analyze_frame(frame, model) -> None:
    results = model(frame, verbose=False)
    
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{class_name} {round(confidence * 100)}%"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"DETECTED: {class_name} — {round(confidence * 100)}% confident")
    
    cv2.imshow("Backyard Monitor", frame)

def main() -> None:
    while True:
        frame = capture_a_frame(cap)
        if frame is not None:
            analyze_frame(frame, model)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()