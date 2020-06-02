import numpy as np
import torch 
def compute_iou(box1, box2, wh= False):
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int((box1[0]- box1[2])/2.0), int((box1[1]- box1[3])/2.0)
        xmax1, ymax1 = int((box1[0]+ box1[2])/2.0), int((box1[1]+box1[3])/2.0)
        xmin2, ymin2 = int((box2[0]- box2[2])/2.0), int((box2[1]+box2[3])/2.0)
        xmax2, ymax2 = int((box2[0]+box2[2])/2.0), int((box2[1]+box2[3])/2.0)
        
    xx1 = np.max([xmin1, xmin2])0
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.max([xmax1, xmax2])
    yy2 = np.max([ymax1, ymax2])
    
    area1 = (xmax1 - xmin1)*(ymax1 - ymin1)
    area2 = (xmax2 - xmin2)*(ymax2 - ymin2)
    
    inter_area = (np.max([0, xx2-xx1]))* np.max([0, yy2-yy1])
    iou = inter_area / (area1 + area2 - inter_area +1e-6)
    return iou

def py_cpu_nms(dets, thresh):
     #dets某个类的框，x1、y1、x2、y2、以及置信度score
    #eg:dets为[[x1,y1,x2,y2,score],[x1,y1,y2,score]……]]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    #面積
    areas = (x2- x1 +1)*(y2- y1 +1)
    order = scores.argsort()[::-1]#按照置信度降序排序
    keep = []
    
    while order.size() >0:
        i = order[0]#保留得分最高的
        keep.append(i)
         #得到相交区域,左上及右下 
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
         #得到相交区域,左上及右下 
        w = np.maximum(0.0, xx2-xx1 +1)
        h = np.maximum(0.0, yy2-yy1 +1)
        
        inter = w*h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积） 
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留IoU小于阈值的box 
        inds = np.where(ovr <= thresh)[0]
        order = order[inds+1]#因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位 
        
    return keep                       

def iou(self, box1, box2):
    N = box1.size(0)
    M = box2.size(0)
    
    lt = torch.max(  # 左上角的点
            box1[:, :2].unsqueeze(1).expand(N, M, 2),   # [N,2]->[N,1,2]->[N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),   # [M,2]->[1,M,2]->[N,M,2]
            )
 
    rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),
            )
    wh = rb -lt
    wh[wh < 0] = 0
    
    inter = wh[:,:,0]*wh[:,:,1]
    area1 = (box1[:,2] - box1[:,0])*(box1[:,3] - box1[:,1])
    area2 = (box2[:,2] - box2[:,0])*(box2[:,3] - box2[:,1])
    area1 = area1.unsqueeze(1).expand(N, M)
    area2 = area2.unsqueeze(1).expand(N, M)
    
    iou = inter / (area1 + area2 - inter)
    
    return iou 
    
def nms(self, bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    keep =[]
    while order.numel()>0:
        if order.numel()==1:
            i = order.item()
            keep.append(i)
        else:
            i =order[0].item()
            keep.append(i)
            xx1 = x1[order[1:]].clamp(min= x1[i])
            yy1 = y1[order[1:]].clamp(min= y1[i])
            xx2 = x2[order[1:]].clamp(min= x2[i])
            yy2 = y2[order[1:]].clamp(min= y2[i])
            
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            
            iou = inter / (areas[i] + areas[i] -inter)
            idx = (iou < threshold).nonzero().squeeze()
            
            order = order[idx +1]
    return torch.LongTensor(keep)        
                   
    
    