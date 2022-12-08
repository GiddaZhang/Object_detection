from YOLO.yolo import yolo
import cv2
import time

if __name__ == '__main__':
    start = time.time()

    Yolo = yolo('YOLO\yolov3.txt',
                'YOLO\yolov3.weights',
                'YOLO\yolov3.cfg')
    
    img = cv2.imread('test_IMG_14_02.jpg')
    ans = Yolo.predict(img)

    print((time.time() - start))
    cv2.imwrite('result.jpg', ans)