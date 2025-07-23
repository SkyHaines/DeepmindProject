import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("fail")
else:
    ret, frame = cap.read()
    if ret:
        print("yay")
    else:
        print("fail 2 :(")
cap.release()