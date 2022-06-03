from pydoc import classname
import cv2
import time
import numpy as np
import argparse

# add the arguments for the script
parser = argparse.ArgumentParser(description='Tensorflow Object Detection')
parser.add_argument('-i', "--input", help='input video file', required=True)
args = parser.parse_args()

cap = cv2.VideoCapture('input/' + args.input)

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

def loadTensor():
    with open('../files/object_detection_classes_coco.txt', 'r') as f:
        class_names= f.read().split('\n')

    # get a different color array for each of the classes
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # load the DNN model
    mobile_net_model = cv2.dnn.readNet(model='../files/frozen_inference_graph.pb',
                            config='../files/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                            framework='TensorFlow')
    return class_names, COLORS, mobile_net_model

loadingAttributes = loadTensor()
classname_tensor = loadingAttributes[0]
COLORS = loadingAttributes[1]
mobile_net_tensor = loadingAttributes[2]

out = cv2.VideoWriter('output/video_result_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

def analizeTensor(image):
    image_height, image_width, _ = image.shape
    # create blob from image de 300*300 siempre -- espera una imagen de 3 canales
    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123))
    # create blob from image
    mobile_net_tensor.setInput(blob)
    # forward pass through the mobile_net_model to carry out the detection
    output = mobile_net_tensor.forward()
    for detection in output[0, 0, :, :]:
    # extract the confidence of the detection
        confidence = detection[2]
        # draw bounding boxes only if the detection confidence is above...
        # ... a certain threshold, else skip
        if confidence > .4:
            # get the class id
            class_id = detection[1]
            # map the class id to the class
            class_name = classname_tensor[int(class_id)-1]
            color = COLORS[int(class_id)]
            # get the bounding box coordinates
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            # get the bounding box width and height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            # draw a rectangle around each detected object
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
            # put the FPS text on top of the frame
            
            cv2.putText(image, class_name + " "+ str(confidence), (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image

#detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        image = frame
        newFrame = analizeTensor(image)
        cv2.imshow('image', newFrame)
        out.write(newFrame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()