import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import config
from mainsys import VideoStream

class Detect: 
    def __init__(self):
        # Load config vals
        pkg = importlib.util.find_spec('tflite_runtime')
        use_TPU = config.use_TPU
        GRAPH_NAME = config.GRAPH_NAME
        MODEL_NAME = config.MODEL_NAME
        min_conf_threshold = config.min_conf_threshold
        imW, imH = config.imW, config.imH
        
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_TPU:
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite' 
        CWD_PATH = os.getcwd()
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

        # Load the Tensorflow Lite model.
        if use_TPU:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        #print(PATH_TO_CKPT)
        else:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT)

        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        self.outname = self.output_details[0]['name']

        if ('StatefulPartitionedCall' in self.outname): # This is a TF2 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        else: # This is a TF1 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2
            
        config.interpreter = self.interpreter
        self.vs = config.videostream
        while self.vs is None:
            self.vs = config.videostream
            print("detect - vs.read is none")
        
        
        return
        
    def run(self):
        while True:
            # Get pre-processing time for fps calc
            initial_time = time.time()
            
            frame1 = self.vs.read()
            
            # Detection
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
            input_data = np.expand_dims(frame_resized, axis=0)
            
            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
            self.interpreter.invoke()
            
            # Save detection results
            config.boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            config.classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0] # Class index of detected objects
            config.scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0] # Confidence
        
            # Get elapsed time for fps calc
            elapsed_time = time.time() - initial_time
            config.fps = 1 / elapsed_time
        
        return     
    
    # could probably inherit this from a superclass
    def add_to_parser(self, parser):
        parser.add_argument('--detectdir', help='Specify detection file',default='detect.py')
