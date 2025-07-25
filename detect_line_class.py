import kbSingleton
import cv2
import numpy as np

# There is a decent amount of duplicated code across detect functions, perhaps this should
# be inherited from an absrtact class.

class DetectLine():
    def __init__(self):
        kb = kbSingleton.kb_instance
        self.vs = kb.get('videostream')
        return
    
    #def add_to_parser(self, parser):
        #parser.add_argument('--detectdir', help='Specify detection file',default='detect.py')
        
    def run(self):
        kb = kbSingleton.kb_instance
        while True:
            if self.vs is None:
                print("detect - vs.read is none")
                self.vs = kb.get('videostream')
                continue
            frame1 = self.vs.read()
            frame = frame1.copy()
            
            # Crop for lower half of the image, as any markings on the floor should be in this section of the view
            offset = frame.shape[0] // 2
            alt_frame = frame[offset:, :]
            
            #Detect lines option 1
            alt_frame= cv2.cvtColor(alt_frame, cv2.COLOR_RGB2GRAY)
            alt_frame = cv2.Canny(alt_frame, 50, 150)
            
            #Detect lines option 2 ???
            lines = cv2.HoughLinesP(alt_frame, 1, np.pi/180, threshold=30, minLineLength=40, maxLineGap=50)
            
            # find centre?
            screen_center = frame.shape[1] // 2
            min_dist = float('inf')
            closest_line = None
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        mid_line = (x1 + x2) // 2
                        dist_to_center = abs(mid_line-screen_center)
                        if dist_to_center < min_dist:
                            min_dist = dist_to_center
                            closest_line = (x1, y1, x2, y2)
            
            if closest_line is not None:
                kb.store('closest_line', closest_line)
        return