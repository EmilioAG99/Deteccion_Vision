from pydoc import classname
import cv2
import time
import numpy as np

cap = cv2.VideoCapture('input/video_1.mp4')

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 

# create the `VideoWriter()` object
def loadCaffe():
    with open('../files/object_detection_classes_coco.txt', 'r') as f:
        class_names= f.read().split('\n')

    # get a different color array for each of the classes
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    # load the DNN model
    mobile_net_model = cv2.dnn.readNet(model='../files/VGG.caffemodel',
                            config='../files/VGG.prototxt', 
                            framework='TensorFlow')
    return class_names, COLORS, mobile_net_model

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

modelo = "caffe"
if(modelo == "tensor"):
    loadingAttributes = loadTensor()
    classname_tensor = loadingAttributes[0]
    COLORS = loadingAttributes[1]
    mobile_net_tensor = loadingAttributes[2]
    
elif(modelo == "yolo"):
    loadingAttributes = loadYolo()
    classname_yolo = loadingAttributes[0]
    COLORS = loadingAttributes[1]
    yolo_model = loadingAttributes[2]
    ln = loadingAttributes[3]
elif(modelo == "caffe"):
    loadingAttributes = loadCaffe()
    classname_caffe = loadingAttributes[0]
    COLORS = loadingAttributes[1]
    mobile_net_caffe = loadingAttributes[2]

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
            
def analizeCaffe(image):
    image_height, image_width, _ = image.shape
    # create blob from image de 300*300 siempre -- espera una imagen de 3 canales
    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123))
    # create blob from image
    mobile_net_caffe.setInput(blob)
    # forward pass through the mobile_net_model to carry out the detection
    output = mobile_net_caffe.forward()
    for detection in output[0, 0, :, :]:
    # extract the confidence of the detection
        confidence = detection[2]
        # draw bounding boxes only if the detection confidence is above...
        # ... a certain threshold, else skip
        if confidence > .4:
            # get the class id
            class_id = detection[1]
            # map the class id to the class
            class_name = classname_caffe[int(class_id)-1]
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
        newFrame = analizeCaffe(image)
        cv2.imshow('image', newFrame)
        out.write(newFrame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
             break
    else:
        break

cap.release()
cv2.destroyAllWindows()
