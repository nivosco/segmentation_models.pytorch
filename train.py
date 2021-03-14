import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


DATA_DIR = '/local/data/camvid/'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

ENCODER = 'timm-efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
           'tree', 'signsymbol', 'fence', 'car',
           'pedestrian', 'bicyclist', 'unlabelled']
ACTIVATION = 'softmax'
DEVICE = 'cuda'
EPOCHS = 60
BATCH_SIZE = 16
LR = 3e-4


def get_training_augmentation():
    train_transform = [albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
                shift_limit=0.1, p=1, border_mode=0),
            albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True,
                border_mode=0),
            albu.RandomCrop(height=320, width=320, always_apply=True),
            albu.IAAAdditiveGaussianNoise(p=0.2),
            albu.IAAPerspective(p=0.5),
            albu.OneOf([albu.CLAHE(p=1),albu.RandomBrightness(p=1),albu.RandomGamma(p=1),],p=0.9,),
            albu.OneOf([albu.IAASharpen(p=1),albu.Blur(blur_limit=3,
                p=1),albu.MotionBlur(blur_limit=3, p=1),],p=0.9,),
            albu.OneOf([albu.RandomContrast(p=1),albu.HueSaturationValue(p=1),],p=0.9)]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [albu.PadIfNeeded(384, 480)]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [albu.Lambda(image=preprocessing_fn),albu.Lambda(image=to_tensor,
        mask=to_tensor),]
    return albu.Compose(_transform)


class Dataset(BaseDataset):
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None,):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in
                self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in
                self.ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in
                self.CLASSES]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        return len(self.ids)


print("Create Dataset")
dataset = Dataset(x_train_dir, y_train_dir)

# create segmentation model with pretrained encoder
print("Build model")
model = smp.Unet(
     encoder_name=ENCODER, 
     encoder_weights=ENCODER_WEIGHTS, 
     classes=len(CLASSES), 
     activation=ACTIVATION,)

print("Get preprocessing function")
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

print("Build training dataset")
train_dataset = Dataset(x_train_dir, 
                        y_train_dir, 
                        augmentation=get_training_augmentation(), 
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES,)
valid_dataset = Dataset(x_valid_dir, 
                        y_valid_dir, 
                        augmentation=get_validation_augmentation(), 
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES,)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
        num_workers=4)
loss = smp.utils.losses.JaccardLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5),]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=LR),])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

print("build training precedure")
train_epoch = smp.utils.train.TrainEpoch(
                       model, 
                       loss=loss, 
                       metrics=metrics, 
                       optimizer=optimizer,
                       device=DEVICE,
                       verbose=True,)

valid_epoch = smp.utils.train.ValidEpoch(
                       model, 
                       loss=loss, 
                       metrics=metrics, 
                       device=DEVICE,
                       verbose=True,)


max_score = 0
for i in range(0, EPOCHS):
    print('\nEpoch: {}'.format(i))
    print('LR {}'.format(scheduler.get_lr()))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    scheduler.step()
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved with IoU {}!'.format(max_score))
    if i % EPOCHS == 10:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        print('Reset scheduler')

print("Testing the best model")
best_model = torch.load('./best_model.pth')
test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,)
test_dataloader = DataLoader(test_dataset)
test_epoch= smp.utils.train.ValidEpoch(model=best_model,loss=loss,metrics=metrics,device=DEVICE,)
logs = test_epoch.run(test_dataloader)
print(logs)
