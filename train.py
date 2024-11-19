import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                workers=0,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='',
                )