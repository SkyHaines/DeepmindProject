import config
import cv2
import numpy as np

# There is a decent amount of duplicated code across detect functions, perhaps this should
# be inherited from an absrtact class.

class DetectLine():
    def __init__(self):
        self.vs = config.videostream
        return
    
    def add_to_parser(self, parser):
        parser.add_argument('--detectdir', help='Specify detection file',default='detect.py')
        
    def run(self):
        while True:
            if self.vs is None:
                print("detect - vs.read is none")
                self.vs = config.videostream
                continue
            frame1 = self.vs.read()
            frame = frame1.copy()
            
            # Crop for lower half of the image, as any markings on the floor should be in this section of the view
            offset = frame.shape[0] // 2
            alt_frame = frame[offset:, :]
            
            #Detect lines option 1
            alt_frame= cv2.cvtColor(alt_frame, cv2.COLOR_RGB2GRAY)
            alt_frame = cv2.Canny(alt_frame, 100, 200)
            
            #Detect lines option 2 ???
            config.lines = cv2.HoughLinesP(alt_frame, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=10)
        return