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
    #("action", "Act")
]
GRAPHICS_MODULES = [
    #("graphics_class", "Graphics")
    #("graphics_highlight_line_class", "GraphicsHighlightLine")
]
# --------------------------------------------
        
def initialise(PLUGIN_MODULES, GRAPHICS_MODULES):
    # Define input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='720x480')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    
    def load_plugin(module_path, class_name):
        module = importlib.import_module(module_path)
        instantiated = getattr(module, class_name)
        return instantiated()
    
    # Load & instantiate plugins
    plugins = []
    for MODULE in PLUGIN_MODULES:
        plugin = load_plugin(MODULE[0], MODULE[1])
        plugins.append(plugin)
        plugin.add_parser_params(parser)

    graphics = []
    for MODULE in GRAPHICS_MODULES:
        module = load_plugin(MODULE[0], MODULE[1])
        graphics.append(module)
    
    args = parser.parse_args()
    
    #Initialised knowledge base and store setup knowledge
    kb = kbSingleton.kb_instance
    for key, value in vars(args).items():
        kb.store(key, value)

    return plugins, graphics
 
def main():
    kb = kbSingleton.kb_instance
    plugins, graphics = initialise(PLUGIN_MODULES, GRAPHICS_MODULES)
    
    # Initialize video stream
    videostream = VideoStream(resolution=(kb.get('imW'),kb.get('imH')),framerate=30).start()
    videostream.wait_for_initialise()
    kb.store('videostream', videostream)
    
    # saved frames
    frame_count = 98
    
    freq = cv2.getTickFrequency()
    
    # Initialise plugin threads
    threads = []
    print("Plugins", plugins)
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
        if cv2.waitKey(10) == ord('q'):
            break
        
        #Press 's' to screenshot 
        if cv2.waitKey(30) == 115:
            filename = f"frame_{frame_count:04d}.png"
            filepath = os.path.join('imgs', filename)
            cv2.imwrite(filepath, frame)
            frame_count += 1
            print("Saved frame as: frame_", frame_count)
        
        # inc framecount if uparrow
        if cv2.waitKey(20) == 82:
            frame_count += 1
            print("Increased frame_count to:", frame_count)
            
        # dec framecount if down arrow
        if cv2.waitKey(20) == 84:
            frame_count -= 1
            print("Decreased frame_count to:", frame_count)
        
    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
    
if __name__ == "__main__":
    main()
