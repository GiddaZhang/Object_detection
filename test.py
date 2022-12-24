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

#已经分好块的图像做检测
def predict_blocks(img_blocks,origin_h,origin_w,h,w,num_h,num_w):
     class_ids, confidences, boxes = [], [], []
     for i in range(num_h):
        for j in range(num_w):
            id, conf, box = Yolo.predict(img_blocks[i*num_w+j],0.5,0.1)
            for one_box in box:
                # 切割子图内的坐标+子图在原图中的偏置坐标
                one_box[1] += i*(h//num_h)+origin_h
                one_box[0] += j*(w//num_w)+origin_w
            class_ids.extend(id)
            confidences.extend(conf)
            boxes.extend(box)
     return class_ids, confidences, boxes

#单次切割并检测
def cut_and_predict(img, origin_h,origin_w,num_h, num_w, overlap):
    #img：图片，未切割
    #origin_h，origin_w:如果是整张图的部分图传入，则“原点”就是左上角，之后用来补足偏移，把部分图的坐标转回整张图坐标；
    #                   如果传入的图片是整张图，原点就是（0,0）                 
    img_blocks = img_proc.img_cut(img, num_h, num_w, overlap)
    h, w = img.shape[0], img.shape[1]
    class_ids, confidences, boxes=predict_blocks(img_blocks,origin_h,origin_w,h,w,num_h, num_w)
    return class_ids, confidences, boxes

#对特别大、特别小的框，扩展一个周围区域用YOLO，重新检测
def box_check(class_ids, confidences, boxes):
    sum=0
    for one_box in boxes:
        S=one_box[2]*one_box[3]
        sum=sum+S
    average=sum/len(boxes)

    new_class_ids, new_confidences, new_boxes = [], [], []
    for one_box in boxes:
        if one_box[2]*one_box[3]>4*average:
            temp_class_ids, temp_confidences, temp_boxes=cut_and_predict(img[int(one_box[1]):int(one_box[1]+one_box[3]),
                                                      int(one_box[0]):int(one_box[0]+one_box[2])],
                                                      int(one_box[1]),int(one_box[0]),1,1,1)

            new_class_ids.extend(temp_class_ids)
            new_confidences.extend(temp_confidences)
            new_boxes.extend(temp_boxes)
        elif one_box[2]*one_box[3]<average/8:
            temp_class_ids, temp_confidences, temp_boxes=cut_and_predict(img[int(one_box[1]-5*one_box[3]):int(one_box[1]+5*one_box[3]),
                                                      int(one_box[0]-5*one_box[2]):int(one_box[0]+5*one_box[2])],
                                                      int(one_box[1]-5*one_box[3]),int(one_box[0]-5*one_box[2]),1,1,1)
            new_class_ids.extend(temp_class_ids)
            new_confidences.extend(temp_confidences)
            new_boxes.extend(temp_boxes)
    class_ids.extend(new_class_ids)
    confidences.extend(new_confidences)
    boxes.extend(new_boxes)
    return class_ids, confidences, boxes   

#删除特别大、特别小、特别长、特别扁的框
def box_delete(class_ids, confidences, boxes):
    sum_S,sum_ratio=0,0
    for one_box in boxes:
        S=one_box[2]*one_box[3]
        ratio=one_box[2]/one_box[3]
        sum_S=sum_S+S
        sum_ratio=sum_ratio+ratio
    average_S=sum_S/len(boxes)  
    average_ratio=sum_ratio/len(boxes)    
    
    i=0
    while i<len(boxes):
        if boxes[i][2]*boxes[i][3]>10*average_S or boxes[i][2]*boxes[i][3]<average_S/10 or boxes[i][2]/boxes[i][3]>10*average_ratio or boxes[i][2]/boxes[i][3]<average_ratio/10:
           del(class_ids[i])
           del(confidences[i])
           del(boxes[i])
           continue
        i+=1
    return class_ids, confidences, boxes   
    

if __name__ == '__main__':
    start = time.time()

    Yolo = yolo('YOLO\yolov3.txt',
                'YOLO\yolov3.weights',
                'YOLO\yolov3.cfg')
    img_proc = ImgProcessor()

    img = cv2.imread('test_IMG_14_02.jpg')
    #h, w = img.shape[0], img.shape[1]

    #第一种切割
    #把切割和搜索合并成一个函数cut_and_predict
    num_h, num_w = 8, 8
    overlap=1.1
    class_ids, confidences, boxes = cut_and_predict(img,0,0,num_h, num_w, overlap)

    #第二种切割
    #num_h, num_w = 10, 10
    #overlap=1.1
    #class_ids_2, confidences_2, boxes_2 = cut_and_predict(img,0,0,num_h, num_w, overlap)

    #两次切割结果合并
    #class_ids.extend(class_ids_2)
    #confidences.extend(confidences_2)
    #boxes.extend(boxes_2)

    # nms抑制
    class_ids, confidences, boxes = Yolo.nms(class_ids, confidences, boxes,0.5,0.1)
    # 把大框区域重新搜索
    class_ids, confidences, boxes = box_check(class_ids, confidences, boxes) 
    # 删去结果中特别的框
    class_ids, confidences, boxes = box_delete(class_ids, confidences, boxes) 
    #再做nms
    class_ids, confidences, boxes = Yolo.nms(class_ids, confidences, boxes,0.5,0.1)

    after_nms = draw_predict(img, class_ids, confidences, boxes, Yolo)
    cv2.imwrite('cut_nms_check_delete_nms.jpg', after_nms)

    # 合并框也写成函数img_proc.box_merge了，可以直接输入所有框
    # 但是没啥用。。。。。。
    # class_ids, confidences, boxes = img_proc.box_merge(img, num_h, num_w, overlap, class_ids, confidences, boxes)

    print((time.time() - start))
    