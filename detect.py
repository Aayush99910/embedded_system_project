from typing import Any, Optional
from ultralytics import YOLO

# importing the opencv library 
import cv2 

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

# First I will be taking a screenshot from the camera I will be using my webcam for now
def capture_a_frame(cap) -> Optional[Any]: 
    ret, frame = cap.read() # grabbing the first frame 

    # if we have the picture we will return the frame
    if ret: 
        return frame 
    else:
        print("Camera not working!")
        return None
    

# now a function to analyze the picture that was taken
def analyze_frame(frame, model) -> None:
    results = model(frame, verbose=False)
    print(results)

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
                print(f"DETECTED: {class_name} — {round(confidence * 100)}% confident")
    
    # show the frame with boxes drawn on it
    cv2.imshow("Backyard Monitor", frame)
    cv2.waitKey(0)


def main() -> None:
    model = YOLO("yolov8n.pt") # we are using the yolo v8 nano because they are very lightweight and also fast
    cap = cv2.VideoCapture(0) # we are connecting to the camera 

    # warmup
    # Just throwing away the first 5 images because I was getting black image
    for i in range(5):
        cap.read()

    frame = capture_a_frame(cap)
    analyze_frame(frame, model)


    # after we have captured it we can free the webcam 
    cap.release()



if __name__ == "__main__":
    main()

