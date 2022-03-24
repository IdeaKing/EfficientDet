data_dir = 'data/VOC2012'
image_dir = 'images'
label_dir = 'labels'
phi = 0
epochs = 50
steps = 1000
batch_size = 2
weight_path = "weights/D0/model31.h5" #'imagenet'
image_size = (512, 640, 768, 896, 1024, 1280, 1408)[phi]
classes = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

