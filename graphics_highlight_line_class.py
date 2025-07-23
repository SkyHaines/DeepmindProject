import config
import cv2
import numpy as np
import time

class GraphicsHighlightLine():
    def __init__(self, colour=[255,0,0], thickness=3):
        self.colour = colour
        self.thickness = thickness
        return
    
    def draw(self, frame):
        lines = config.lines
        offset = frame.shape[0] // 2
        if lines is None:
            print("GraphicsHighlightLine -- Detection not ready")
            time.sleep(1)
            return
        img = np.copy(frame)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8,)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1 + offset), (x2, y2 + offset), self.colour, self.thickness)
        img = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0.0)
        return img
    
    