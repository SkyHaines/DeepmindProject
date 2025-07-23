def add_to_parser(parser):
    parser.add_argument('--detectdir', help='Specify detection file',default='detect.py') 

def main():
    import config
    import cv2
    
    frame = config.currentFrame
    if frame is None:
        return
    
    # Crop for lower half of the image, as any markings on the floor should be in this section of the view
    offset = image.shape[0] // 2
    alt_image = image[offset:, :]
    
    #Detect lines option 1
    alt_image = cv2.cvtColor(alt_image, cv2.COLOR_RGB2GRAY)
    alt_image = cv2.Canny(alt_image, 100, 200)
    
    #Detect lines option 2 ???
    config.lines = cv2.HoughLinesP(alt_image, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=10)
    return
    
    