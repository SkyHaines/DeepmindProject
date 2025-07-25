import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util
import config
import threading
from videostream import VideoStream
import kbSingleton

# ------------- MODULE IMPORTS ----------------
# module path, class name
PLUGIN_MODULES = [
    #("detect_class", "Detect"),
    ("detect_line_class","DetectLine"),
    ("action", "Act")
]
GRAPHICS_MODULES = [
    #("graphics_class", "Graphics")
    ("graphics_highlight_line_class", "GraphicsHighlightLine")
]
# --------------------------------------------
        
def initialise(PLUGIN_MODULES, GRAPHICS_MODULES):
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    
    args = parser.parse_args()
    
    #Initialised knowledge base and store setup knowledge
    kb = kbSingleton.kb_instance
    kb.store('MODEL_NAME', args.modeldir)
    kb.store('GRAPH_NAME', args.graph)
    kb.store('LABELMAP_NAME', args.labels)
    kb.store('min_conf_threshold', float(args.threshold))
    kb.store('resW', args.resolution.split('x')[0]),
    kb.store('resH', args.resolution.split('x')[1])
    kb.store('imW', int(kb.get('resW')))
    kb.store('imH', int(kb.get('resH')))
    kb.store('use_TPU', args.edgetpu)
    
    def load_plugin(module_path, class_name):
        module = importlib.import_module(module_path)
        instantiated = getattr(module, class_name)
        return instantiated()
    
    plugins = []
    for MODULE in PLUGIN_MODULES:
        plugin = load_plugin(MODULE[0], MODULE[1])
        plugins.append(plugin)

    graphics = []
    for MODULE in GRAPHICS_MODULES:
        module = load_plugin(MODULE[0], MODULE[1])
        graphics.append(module)
    
    return plugins, graphics
 
def main():
    kb = kbSingleton.kb_instance
    plugins, graphics = initialise(PLUGIN_MODULES, GRAPHICS_MODULES)
    
    # Initialize video stream
    videostream = VideoStream(resolution=(kb.get('imW'),kb.get('imH')),framerate=30).start()
    videostream.wait_for_initialise()
    kb.store('videostream', videostream)
    
    freq = cv2.getTickFrequency()
    
    # Initialise plugin threads
    threads = []
    for plugin in plugins:
        #plugin_instance = plugin
        thread = threading.Thread(target=plugin.run, daemon=True)
        thread.start()
        threads.append(thread)
    
    # Main loop
    while True:
        # Allow all graphics modules to draw onto the frame prior to displaying it
        frame = videostream.read().copy()
        for module in graphics:
            frame = module.draw(frame)
        if frame is not None:
            cv2.imshow('Object detector', frame)        
        
        #Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        
    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
    
if __name__ == "__main__":
    main()