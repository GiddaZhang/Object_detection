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


def box_merge(block_A,block_B):
    #对AB两区域中的boxes逐一比对判断是否合并
    i=0
    while i<len(block_A[0]):
       j=0
       while j<len(block_B[0]):
          if block_A[0][i]!=block_B[0][j]:   #box的种类不同，跳过
            j=j+1
            continue
          else:    #box的种类相同，计算是否有交集
            #两角点位置信息储存box
            box_A=(block_A[2][i][0],block_A[2][i][1],block_A[2][i][0]+block_A[2][i][2],block_A[2][i][1]+block_A[2][i][3])
            box_B=(block_B[2][j][0],block_B[2][j][1],block_B[2][j][0]+block_B[2][j][2],block_B[2][j][1]+block_B[2][j][3])
            #两box面积
            S_box_A=block_A[2][i][2]*block_A[2][i][3]
            S_box_B=block_B[2][j][2]*block_B[2][j][3]
            #求相交“矩形” I
            boxI_x1 = max(min(box_A[0],box_A[2]),min(box_B[0],box_B[2]))   ##相交矩形的横坐标最小值
            boxI_x2 = min(max(box_A[0],box_A[2]),max(box_B[0],box_B[2]))   ##相交矩形的横坐标最大值
            boxI_y1 = max(min(box_A[1],box_A[3]),min(box_B[1],box_B[3]))   ##相交矩形的纵坐标最小值
            boxI_y2 = min(max(box_A[1],box_A[3]),max(box_B[1],box_B[3]))   ##相交矩形的纵坐标最大值
            #if (boxI_x2>=boxI_x1 and boxI_y2>boxI_y1)or(boxI_x2>boxI_x1 and boxI_y2>=boxI_y1):   #矩形相交(面、线)，合并box
            if (boxI_x2>boxI_x1 and boxI_y2>boxI_y1):   #矩形相交(面)，边界
                #新的box边界，取并矩形U，用xywh形式储存
                boxU_x = min(box_A[0],box_A[2],box_B[0],box_B[2])   ##
                boxU_y = min(box_A[1],box_A[3],box_B[1],box_B[3])   ##
                boxU_w = max(box_A[0],box_A[2],box_B[0],box_B[2])-min(box_A[0],box_A[2],box_B[0],box_B[2])
                boxU_h = max(box_A[1],box_A[3],box_B[1],box_B[3])-min(box_A[1],box_A[3],box_B[1],box_B[3])   ##
                #面积加权置信度
                boxU_conf=(block_A[1][i]*S_box_A+block_B[1][j]*S_box_B)/(S_box_A+S_box_B)    
                if S_box_A>S_box_B:
                    #如果框在A面积更大，将新的box归入A区，改A删B
                    block_A[1][i]=boxU_conf
                    block_A[2][i][0]=boxU_x
                    block_A[2][i][1]=boxU_y
                    block_A[2][i][2]=boxU_w
                    block_A[2][i][3]=boxU_h
                    del(block_B[0][j])
                    del(block_B[1][j])
                    del(block_B[2][j])
                else:
                    #将新的box归入B区，改B删A
                    block_B[1][j]=boxU_conf
                    block_B[2][j][0]=boxU_x
                    block_B[2][j][1]=boxU_y
                    block_B[2][j][2]=boxU_w
                    block_B[2][j][3]=boxU_h
                    del(block_A[0][i])
                    del(block_A[1][i])
                    del(block_A[2][i])
                    i=i-1
                    break
            else:
              j=j+1
              continue
       i=i+1 
    return block_A,block_B


if __name__ == '__main__':
    start = time.time()

    Yolo = yolo('YOLO\yolov3.txt',
                'YOLO\yolov3.weights',
                'YOLO\yolov3.cfg')
    
    img = cv2.imread('test_IMG_14_02.jpg')
    h, w = img.shape[0], img.shape[1]
    num_h, num_w = 10, 10
    img_blocks = img_cut(img, num_h, num_w, 1.1)

    #把检测结果按各个小块存储
    blocks_output=[]
    for i in range(num_h):
        for j in range(num_w):
            id,conf,box = Yolo.predict(img_blocks[i*num_w+j])
            for one_box in box:
                # 切割子图内的坐标+子图在原图中的偏置坐标
                one_box[1] += i*(h//num_h)
                one_box[0] += j*(w//num_w)
            blocks_output.append((id,conf,box))
    
    #每一小块和右侧、下方合并
    num_blocks=num_h*num_w
    for i in range(num_blocks):
        if(i+1>=0)and(i+1<num_blocks):
            blocks_output[i],blocks_output[i+1]=box_merge(blocks_output[i],blocks_output[i+1])
        if(i+num_w>=0)and(i+num_w<num_blocks):
            blocks_output[i],blocks_output[i+num_w]=box_merge(blocks_output[i],blocks_output[i+num_w])

    #整合结果
    class_ids=[]
    confidences=[]
    boxes=[]
    for i in range(num_blocks):
        class_ids.extend(blocks_output[i][0])
        confidences.extend(blocks_output[i][1])
        boxes.extend(blocks_output[i][2])

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

    cv2.imwrite('result.jpg', img)
    
    print((time.time() - start))
    