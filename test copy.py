from YOLO.yolo import yolo
import numpy as np
import cv2
import time
import json

def img_cut(img, num_h, num_w, overlap):

    h, w = img.shape[0], img.shape[1]           # 原图高度，宽度
    h_block, w_block = h // num_h, w // num_w   # 切割图像高度，宽度
    h_block_ovlap = round(overlap * h_block)
    w_block_ovlap = round(overlap * w_block)
    img_blocks = []

    for i in range(num_h):
        for j in range(num_w):
            if i is num_h - 1 and j is num_w - 1:
                img_block = img[i*h_block:(i+1)*h_block, j*w_block:(j+1)*w_block]
            elif i is num_h - 1 and j is not num_w - 1:
                img_block = img[i*h_block:(i+1)*h_block, j*w_block:(j+1)*w_block_ovlap]
            elif i is not num_h - 1 and j is num_w - 1:
                img_block = img[i*h_block:(i+1)*h_block_ovlap, j*w_block:(j+1)*w_block]
            else:
                img_block = img[i*h_block:(i+1)*h_block_ovlap, j*w_block:(j+1)*w_block_ovlap]
            img_blocks.append(img_block)

    return img_blocks
    
def img_comb(img_list, num_h, num_w):

    h, w = img_list[0].shape[0], img_list[0].shape[1]
    img = np.zeros([h*num_h, w*num_w, 3], np.uint8)
    for i in range(num_h):
        for j in range(num_w):
            img[i*h:(i+1)*h, j*w:(j+1)*w] = img_list[j+num_w*i]
    
    return img

if __name__ == '__main__':
    start = time.time()

    pb_file = 'frozen_inference_graph.pb' #http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
    pbtxt_file = 'graph.pbtxt'
    
    img = cv2.imread('test_IMG_14_02.jpg')

    score_threshold = 0.3

    try:
        net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
    except:
        net = cv2.dnn.readNetFromTensorflow('Faster_RCNN/'+pb_file, 'Faster_RCNN/'+pbtxt_file)
    h, w = img.shape[0], img.shape[1]
    num_h, num_w = 20, 20
    img_blocks = img_cut(img, num_h, num_w, 1.1)
    class_ids, confidences, boxes = [], [], []
    for i in range(num_h):
        for j in range(num_w):
            id, conf, box = Yolo.predict(img_blocks[i*num_w+j])
            for one_box in box:
                one_box[1] += i*(h//num_h)
                one_box[0] += j*(w//num_w)
            class_ids.extend(id)
            confidences.extend(conf)
            boxes.extend(box)
    
    for i in range(len(boxes)):
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        Yolo.draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))  #画框、标类
    cv2.imwrite('result/2.jpg', img)

    img = cv2.imread('test_IMG_14_02.jpg')
    class_ids, confidences, boxes = Yolo.nms(class_ids, confidences, boxes)
    for i in range(len(boxes)):
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        Yolo.draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))  #画框、标类

    cv2.imwrite('result/1.jpg', img)
    
    print((time.time() - start))
    