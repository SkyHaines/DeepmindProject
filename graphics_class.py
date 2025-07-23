import config
import cv2
import os

class Graphics():
    
    def __init__(self):
        
        # Find and store labelmap to config
        CWD_PATH = os.getcwd()
        PATH_TO_LABELS = os.path.join(CWD_PATH,config.MODEL_NAME,config.LABELMAP_NAME)
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        if labels[0] == '???':
            del(labels[0])
        config.labels = labels
        
        return
    
    def draw(self, frame):
        # Retrieve data to draw with
        interpreter = config.interpreter
        if interpreter is None:
            print("Interpreter not ready")
            return
        output_details = interpreter.get_output_details()
        boxes = config.boxes
        classes = config.classes
        scores = config.scores
        if scores is None:
            print("Detection not ready")
            return
        min_conf_threshold = config.min_conf_threshold
        resW, resH = config.resW, config.resH
        imW, imH = config.imW, config.imH
        
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                        
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = config.labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(config.fps),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        return frame

