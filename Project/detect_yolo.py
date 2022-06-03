from pydoc import classname
import cv2
import time
import numpy as np
import argparse

# add the arguments for the script
parser = argparse.ArgumentParser(description='YOLO Object Detection')
parser.add_argument('-i', "--input", help='input video file', required=True)
args = parser.parse_args()

cap = cv2.VideoCapture('input/' + args.input)

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

def loadYolo():
    with open('../files/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

# get a different color array for each of the classes
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # load the DNN model
    yolo_model = cv2.dnn.readNetFromDarknet('../files/yolov3.cfg', '../files/yolov3.weights')

    ln = yolo_model.getLayerNames()
    ln = [ln[i-1] for i in yolo_model.getUnconnectedOutLayers()]
    return class_names, COLORS, yolo_model, ln

loadingAttributes = loadYolo()
classname_yolo = loadingAttributes[0]
COLORS = loadingAttributes[1]
yolo_model = loadingAttributes[2]
ln = loadingAttributes[3]

out = cv2.VideoWriter('output/video_result_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

def analizeYolo(image):
    image_height, image_width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), crop=False)
    yolo_model.setInput(blob)

    layerOutputs = yolo_model.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []


    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.3:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([image_width, image_height,image_width, image_height])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(round(float(confidence),2))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{classname_yolo[classIDs[i]]}"
            cv2.putText(image, text +" "+ str(confidences[i]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

#detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        image = frame
        newFrame = analizeYolo(image)
        cv2.imshow('image', newFrame)
        out.write(newFrame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()