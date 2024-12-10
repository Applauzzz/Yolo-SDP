#coding:utf-8
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/mnt/nvme_storage/Zehao/ultralytics_improve - SPD-Conv/ultralytics/cfg/models/v8/yolov8-cls.yaml')
    model.load('yolov8n-cls.pt') # loading pretrain weights
    model.train(data='/mnt/nvme_storage/database/mine-site/data', epochs=250, 
    imgsz=640, 
    batch=70,
    lr0=0.0001, 
    lrf=0.025,    
    device="0,1,2,3,4,5"             
)
