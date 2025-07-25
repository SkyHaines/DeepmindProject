import kbSingleton
import cv2
import numpy as np
import time

class GraphicsHighlightLine():
    def __init__(self, colour=[0,255,0], thickness=5):
        self.colour = colour
        self.thickness = thickness
        return
    
    def draw(self, frame):
        kb = kbSingleton.kb_instance
        closest_line = kb.get('closest_line')
        offset = frame.shape[0] // 2
        if closest_line is None:
            print("GraphicsHighlightLine -- Detection not ready")
            time.sleep(1)
            return
        img = np.copy(frame)
        
        # Creates a transparent image of the desired shape.
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8,)
        
        # Draws line highlight onto the empty image.
        for x1, y1, x2, y2 in [closest_line]:
            cv2.line(line_img, (x1, y1 + offset), (x2, y2 + offset), self.colour, self.thickness)
        
        # Adds circle to middle of frame for reference 
        cv2.circle(line_img, (frame.shape[1]//2,(3*frame.shape[0])//4), radius=5, color=[0,0,255], thickness=15)
        
        # Overlays transparent image with highlights onto original frame
        img = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0.0)
        
        return img
    
    