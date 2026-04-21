from typing import Any, Optional
from ultralytics import YOLO
from picamera2 import Picamera2

# importing the opencv library 
import cv2

# importing the time 
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
    #cv2.imshow("Backyard Monitor", frame)
    #cv2.waitKey(0)
    
    cv2.imwrite("lastcapture.jpg", frame)
    print("Saved to lastcapture.jpg")

def main() -> None:
    start = time.time()

    model = YOLO("yolov8n.pt") # we are using the yolo v8 nano because they are very lightweight and also fast
    picam2 = Picamera2()# we are connecting to the camera 

    # we are setting up the camera configuration.
    #config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    config = picam2.create_video_configuration(raw={"size": (1640, 1232)}, main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start() # and here we are turning the camera on and starting to capture the image

    # warmup
    # Sleep for 2 seconds  because I was getting black image
    #time.sleep(2)

    frame = capture_a_frame(picam2)
    analyze_frame(frame, model)


    # after we have captured it we can free the camera
    picam2.stop()
    end = time.time()

    # total time it took 
    print(f"Total time that it took: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
