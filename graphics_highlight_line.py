import kbSingleton
import cv2

def main():
    kb = kbSingleton.kb_instance
    return highlightLines(kb.get('currentFrame'), lines)
    
def highlightLines(image, lines, colour=[255,0,0], thickness=3):
    img = np.copy(image)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8,)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1 + offset), (x2, y2 + offset), colour, thickness)
    img = cv2.addWeighted(image, 0.8, line_img, 1.0, 0.0)
    return img