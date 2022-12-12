from YOLO.yolo import yolo
import numpy as np
import cv2
import time
import json

def img_cut(img, num_h, num_w, overlap):
    # img:图像
    # num_h:高度方向切分数量 
    # num_w:宽度方向切分数量 
    # overlap:重叠区域占比
    # 返回num_h*num_w张切割图像
    h, w = img.shape[0], img.shape[1]           # 原图高度，宽度
    h_block, w_block = h // num_h, w // num_w   # 切割图像高度，宽度
    h_block_ovlap = round(overlap * h_block)    # 考虑重叠区域后的高度
    w_block_ovlap = round(overlap * w_block)    # 考虑重叠区域后的宽度

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

if __name__ == '__main__':
    start = time.time()

    Yolo = yolo('YOLO\yolov3.txt',
                'YOLO\yolov3.weights',
                'YOLO\yolov3.cfg')
    
    img = cv2.imread('test_IMG_14_02.jpg')
    h, w = img.shape[0], img.shape[1]
    num_h, num_w = 10, 10
    img_blocks = img_cut(img, num_h, num_w, 1.1)

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
    
    # 下面是比对不做非极大值抑制之前的图像
    # for i in range(len(boxes)):
    #     box = boxes[i]
    #     x, y, w, h = box[0], box[1], box[2], box[3]
    #     Yolo.draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))  #画框、标类
    # cv2.imwrite('result/yolo/2.jpg', img)

    img = cv2.imread('test_IMG_14_02.jpg')
    class_ids, confidences, boxes = Yolo.nms(class_ids, confidences, boxes)
    for i in range(len(boxes)):
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        Yolo.draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))  #画框、标类

    cv2.imwrite('result/yolo/1.jpg', img)
    
    print((time.time() - start))
    