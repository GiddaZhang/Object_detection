from YOLO.yolo import yolo
from ImgProcessor import ImgProcessor
import numpy as np
import cv2
import time
import json

def draw_predict(img, class_ids, confidences, boxes, Yolo):
    tmp = img.copy()
    for i in range(len(boxes)):
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        Yolo.draw_prediction(tmp, class_ids[i], confidences[i], round(
            x), round(y), round(x+w), round(y+h))
    return tmp


if __name__ == '__main__':
    start = time.time()

    Yolo = yolo('YOLO\yolov3.txt',
                'YOLO\yolov3.weights',
                'YOLO\yolov3.cfg')
    img_proc = ImgProcessor()

    img = cv2.imread('test_IMG_14_02.jpg')
    h, w = img.shape[0], img.shape[1]
    num_h, num_w = 7, 7
    overlap=1.1
    img_blocks = img_proc.img_cut(img, num_h, num_w, overlap)

    # 把检测结果按各个小块存储
    class_ids, confidences, boxes = [], [], []
    for i in range(num_h):
        for j in range(num_w):
            id, conf, box = Yolo.predict(img_blocks[i*num_w+j])
            for one_box in box:
                # 切割子图内的坐标+子图在原图中的偏置坐标
                one_box[1] += i*(h//num_h)
                one_box[0] += j*(w//num_w)
            class_ids.extend(id)
            confidences.extend(conf)
            boxes.extend(box)

    # 检测原图
    origin = draw_predict(img, class_ids, confidences, boxes, Yolo)
    cv2.imwrite('result1.jpg', origin)

    # nms抑制
    class_ids, confidences, boxes = Yolo.nms(class_ids, confidences, boxes)
    after_nms = draw_predict(img, class_ids, confidences, boxes, Yolo)
    cv2.imwrite('result2.jpg', after_nms)

    # 合并框
    img = cv2.imread('test_IMG_14_02.jpg')
    class_ids, confidences, boxes = img_proc.box_merge(img, num_h, num_w, overlap, class_ids, confidences, boxes)
    class_ids, confidences, boxes = Yolo.nms(class_ids, confidences, boxes)
    after_merge = draw_predict(img, class_ids, confidences, boxes, Yolo)
    cv2.imwrite('result3.jpg', after_merge)


    print((time.time() - start))
    