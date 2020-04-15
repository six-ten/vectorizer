import cv2
import numpy as np 

img = cv2.imread('./imgs/instadp.jpg')

frame = cv2.imread('./imgs/segmenter.jpg')




while True :
    cv2.imshow("original",cv2.medianBlur(frame,7))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("filtered",gray)
    edged = cv2.Canny(gray, 0, 255)
    cv2.imshow("Canny",edged)
    
    if cv2.waitKey(1) == 27 :
        break

cv2.destroyAllWindows()
