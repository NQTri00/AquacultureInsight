import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/DGF-RT-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='DGF-RT-DETR\dataset\dataset/snail.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8, #                 
                workers=4, 
                # device='0', 
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )