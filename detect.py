import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/train/rtdetr-timm/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/images/train',
                  project='runs/detect',
                  name='MADL',
                  save=True,
                  #visualize=True # visualize model features maps
                  )