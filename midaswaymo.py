
import sys
import yaml
sys.path.append('/ari/users/ibaskaya/projeler/midaswaymo')
from smallmidas import MidasNet_small

from mappings import sq_w2k_safe, sq_k2w_safe


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict
from glob import glob
from scipy.ndimage import convolve
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from semutilsmine import fast_interpolate
DEBUG=False


import random
import torch
from torchvision import transforms

class FastFill:
    def __init__(self, p=1):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            return image, label

        rangeimg = image[0,...]
        intensity = image[-1,...]
        
        rangeimg = fast_interpolate(rangeimg)
        intensityimg = fast_interpolate(intensityimg)
        img = np.stack((rangeimg,intensity),axis=0)
        return img, label

class RandomRescaleRangeImage:
    def __init__(self, scale_range=(1, 1.5), p=0.05):
        """
        Initializes the RandomRescaleRangeImage transform.
        
        Parameters:
        - scale_range: Tuple of (min_scale, max_scale) for random scaling.
        """
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image, label):
        """
        Applies random rescaling to both the image and segmentation label.

        Parameters:
        - image: The input image tensor of shape (C, H, W).
        - label: The input segmentation label tensor of shape (H, W).

        Returns:
        - rescaled_image: The rescaled image tensor.
        - rescaled_label: The rescaled segmentation label tensor.
        """
        if random.random() < self.p:
            return image, label

        # Randomly select a scaling factor
        scale = random.uniform(*self.scale_range)
        
        # Get original size
        original_size = image.shape[-2:]  # (H, W)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        
        # Rescale the image dimensions (with interpolation)
        rescaled_image = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST)(torch.tensor(image))
        
        # Rescale the segmentation label (with nearest neighbor interpolation)
        rescaled_label = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST)(torch.tensor(label).unsqueeze(0))

        # Adjust the pixel values of the image based on the scale
        mask = rescaled_image != -1
        rescaled_image[mask] = rescaled_image[mask] / scale

        return rescaled_image.numpy(), rescaled_label.squeeze().numpy()



def calculate_miou(preds, labels, num_classes=23):
    iou_sum = 0.0
    valid_classes = 0

    for cls in range(1, num_classes):
        intersection = torch.logical_and(preds == cls, labels == cls).sum().item()
        union = torch.logical_or(preds == cls, labels == cls).sum().item()
        if union > 0:
            iou_sum += intersection / union
            valid_classes += 1

    return iou_sum / valid_classes

def calculate_classwise_intersection_union(preds, labels, num_classes=23):
    classwise_intersection = torch.zeros(num_classes, dtype=torch.float32)
    classwise_union = torch.zeros(num_classes, dtype=torch.float32)
    
    for cls in range(0, num_classes):  # Ignore class 0 (usually background)
        intersection = torch.logical_and(preds == cls, labels == cls).sum().item()
        union = torch.logical_or(preds == cls, labels == cls).sum().item()
        
        classwise_intersection[cls] = intersection
        classwise_union[cls] = union

    return classwise_intersection, classwise_union

def calculate_final_miou_from_batches(batch_results, num_classes=23):

    # Initialize accumulators for the total intersection and union
    total_classwise_intersection = torch.zeros(num_classes, dtype=torch.float32)
    total_classwise_union = torch.zeros(num_classes, dtype=torch.float32)

    # Accumulate intersections and unions across all batches
    for classwise_intersection, classwise_union in batch_results:
        total_classwise_intersection += classwise_intersection
        total_classwise_union += classwise_union

    # Now compute the IoUs
    classwise_iou = []
    valid_classes = 0
    iou_sum = 0.0

    for cls in range(1, num_classes):  # Ignore class 0 (usually background)
        intersection = total_classwise_intersection[cls]
        union = total_classwise_union[cls]
        
        if union > 0:
            iou = intersection / union
            classwise_iou.append(iou.item())
            iou_sum += iou.item()
            valid_classes += 1
        else:
            classwise_iou.append(float('nan'))  # No instance of this class

    # Calculate mean IoU (ignoring NaN classes)
    mean_iou = iou_sum / valid_classes if valid_classes > 0 else float('nan')

    # Calculate total IoU (total intersection over total union)
    total_intersection = total_classwise_intersection.sum().item()
    total_union = total_classwise_union.sum().item()
    total_iou = total_intersection / total_union if total_union > 0 else float('nan')

    return classwise_iou, mean_iou, total_iou

def print_miou_results(classwise_iou, mean_iou, total_iou):
    waymo = {
        0: 'TYPE_UNDEFINED',
        1: 'TYPE_CAR',
        2: 'TYPE_TRUCK',
        3: 'TYPE_BUS',
        4: 'TYPE_OTHER_VEHICLE',
        5: 'TYPE_MOTORCYCLIST',
        6: 'TYPE_BICYCLIST',
        7: 'TYPE_PEDESTRIAN',
        8: 'TYPE_SIGN',
        9: 'TYPE_TRAFFIC_LIGHT',
        10: 'TYPE_POLE',
        11: 'TYPE_CONSTRUCTION_CONE',
        12: 'TYPE_BICYCLE',
        13: 'TYPE_MOTORCYCLE',
        14: 'TYPE_BUILDING',
        15: 'TYPE_VEGETATION',
        16: 'TYPE_TREE_TRUNK',
        17: 'TYPE_CURB',
        18: 'TYPE_ROAD',
        19: 'TYPE_LANE_MARKER',
        20: 'TYPE_OTHER_GROUND',
        21: 'TYPE_WALKABLE',
        22: 'TYPE_SIDEWALK'
    }

    print("Classwise IoU:")
    for i, iou in enumerate(classwise_iou, start=1):  # Start at 1 to skip class 0
        class_name = waymo.get(i, f"Class {i}")
        print(f"  {class_name}: {iou:.4f}" if not isinstance(iou, float) or not iou != iou else f"  {class_name}: N/A")

    print(f"\nMean IoU: {mean_iou:.4f}")
    print(f"Total IoU: {total_iou:.4f}")

# mIoU calculation utility
def calculate_miou(preds, labels, num_classes, ignore_index=0):
    """Calculates mean IoU, ignoring the specified class index (like background)."""
    iou_per_class = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        true_positive = ((preds == cls) & (labels == cls)).sum().item()
        false_positive = ((preds == cls) & (labels != cls)).sum().item()
        false_negative = ((preds != cls) & (labels == cls)).sum().item()
        
        denominator = true_positive + false_positive + false_negative
        if denominator == 0:
            iou = float('nan')  # Skip this class if there's no presence
        else:
            iou = true_positive / denominator
        
        iou_per_class.append(iou)
    
    # Filter out any nan values and compute mean IoU
    iou_per_class = [iou for iou in iou_per_class if not torch.isnan(torch.tensor(iou))]
    if len(iou_per_class) == 0:
        return 0.0  # No classes to calculate IoU
    else:
        return sum(iou_per_class) / len(iou_per_class)

class SegmentationDataset(Dataset):
    def __init__(self, root = '/ari/users/ibaskaya/projeler/lidar-bonnetal/datasets/kitti', split = 'training', 
    transform=None, pretransform=None, fastfill=None, iswaymo=True, width=2656):
        """
        Args:
            parquet_files (list): List of paths to the parquet files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        imagepaths = sorted(glob(os.path.join(root,split,'images','*.npz')))
        labelpaths = [i.replace('images','labels') for i in imagepaths]
        self.datapaths  = [(imagepaths[i], labelpaths[i]) for i in range(len(imagepaths))]
        
        DEBUG = False
        if DEBUG:
            self.datapaths = self.datapaths[:2]
            
        self.transform = transform
        self.pretransform = pretransform
        self.fastfill = fastfill
        self.width = width
        
        """        ARCH = np.array([[16.7921, 15.1336],
                        [ 1.6576, 16.8768],
                        [ 0.8692, 14.5421],
                        [ 0.2838,  3.2056],
                        [-0.0222,  0.4402]])"""

        ARCH = np.array([[20.121, 13.786],
                        [ 1.74, 16.81],
                        [ 1.06, 14.05],
                        [ 0.288,  3.134],
                        [ 0.1443,  0.164]])


        if not iswaymo:
            mapdict = sq_k2w_safe
            max_index = max(mapdict.keys())  # Get the maximum class index from the mapdict
            lookup_tensor = torch.zeros(max_index + 1, dtype=torch.long)  # Create a lookup tensor
            for old_idx, new_idx in mapdict.items():
                lookup_tensor[old_idx] = new_idx
            
            self.lookup_tensor = lookup_tensor
        else:
            self.lookup_tensor = None

        self.iswaymo = iswaymo

        print('Loading data...')

        self.shiftrange, self.scalerange = ARCH[0,0], ARCH[0,1]
        self.shiftintensity, self.scaleintensity = ARCH[4,0], ARCH[4,1]
    
    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, idx):
        imagepath,labelpath = self.datapaths[idx]

        if not self.iswaymo:
            image, label = np.load(imagepath)['image'], np.load(labelpath)['label']
        else:
            image, label = np.load(imagepath)['array'], np.load(labelpath)['array']
            
        image, label = torch.tensor(image, dtype = torch.float32), torch.tensor(label, dtype=torch.long)
        
        if not self.iswaymo:
            label = self.lookup_tensor[label]

        image, label = image.numpy(), label.numpy()

        if self.fastfill:
            image,label = self.fastfill(image,label)

        if self.pretransform:
            image, label = self.pretransform(image, label)
        mask = image[0]==-1
        rangeimage = (image[0,...]-self.shiftrange)/self.scalerange
        intensityimage = (image[-1,...]-self.shiftintensity)/self.scaleintensity
        rangeimage = rangeimage*mask
        intensity = intensityimage*mask
        image = np.stack((rangeimage,intensity))

        if self.transform:
            augmented = self.transform(image=np.transpose(image, (1, 2, 0)), mask=label.astype(np.float32)[...,np.newaxis])
            image = augmented['image'].to(torch.float32)
            label = augmented['mask'].to(torch.long)


        return image, label.squeeze()


# mIoU calculation utility
def calculate_miou(preds, labels, num_classes, ignore_index=0):
    """Calculates mean IoU, ignoring the specified class index (like background)."""
    iou_per_class = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        true_positive = ((preds == cls) & (labels == cls)).sum().item()
        false_positive = ((preds == cls) & (labels != cls)).sum().item()
        false_negative = ((preds != cls) & (labels == cls)).sum().item()
        
        denominator = true_positive + false_positive + false_negative
        if denominator == 0:
            iou = float('nan')  # Skip this class if there's no presence
        else:
            iou = true_positive / denominator
        
        iou_per_class.append(iou)
    
    # Filter out any nan values and compute mean IoU
    iou_per_class = [iou for iou in iou_per_class if not torch.isnan(torch.tensor(iou))]
    if len(iou_per_class) == 0:
        return 0.0  # No classes to calculate IoU
    else:
        return sum(iou_per_class) / len(iou_per_class)

def slide_left(image, slide_fraction=0.1):
    """Slide the image to the left and wrap around."""
    height, width = image.shape[:2]
    shift = int(width * slide_fraction)
    return np.hstack((image[:, shift:], image[:, :shift]))  # Slide left and wrap

if __name__=='__main__':
    model = MidasNet_small()
    #checkpoint = torch.load('model_state_dict.pth')  # Path to your saved model
    #model.load_state_dict(checkpoint)
    pretransform = RandomRescaleRangeImage()
    fastfill = FastFill(p=1)
    transform_train = A.Compose([
        #A.Resize(height=64, width=2650, interpolation=cv2.INTER_NEAREST, p=1),  # Resize
        #A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.0, rotate_limit=0, border_mode=cv2.BORDER_WRAP, p=0.1),  # Shift left with wrap-around effect
        A.RandomCrop(height = 64, width = 2650, p=1),
        A.PadIfNeeded(min_height=64, min_width=2656, border_mode=0, value=0, mask_value=0),
        A.HorizontalFlip(p=0.1),  # Horizontal flip with 20% probability
        #A.CoarseDropout(max_holes=2, max_height=64, max_width=256, min_holes=1, min_height=1, min_width=1, fill_value=0, p=0.1),  # CoarseDropout instead of Cutout
        ToTensorV2()  # Convert to PyTorch tensors
    ], additional_targets={'mask': 'image'})  # Apply same augmentations to mask

    transform_valid = A.Compose([
        A.Resize(height=64, width=2650,interpolation = cv2.INTER_NEAREST,p=1),
        A.PadIfNeeded(min_height=64, min_width=2656, border_mode=0, value=0, mask_value=0),
        #A.RandomCrop(height = 64, width = 1024, p=1),
        ToTensorV2()
    ], additional_targets={'mask': 'image'})

    train_dataset_waymo = SegmentationDataset(root = '/tmp/myadiyaman/waymodataset', 
                                        split = 'training', 
                                        transform=transform_train,
                                        pretransform=pretransform,
                                        fastfill = fastfill,
                                        iswaymo = True)

    validation_dataset_waymo = SegmentationDataset(root = '/tmp/myadiyaman/waymodataset', 
                                        split = 'validation', 
                                        transform=transform_valid,
                                        fastfill = fastfill,
                                        iswaymo = True)

    train_dataset_kitti = None
    """
    train_dataset_kitti = SegmentationDataset(root = '/ari/users/ibaskaya/projeler/lidar-bonnetal/datasets/kitti', 
                                        split = 'training', 
                                        transform=transform_train,
                                        iswaymo = False)
    

    validation_dataset_kitti = SegmentationDataset(root = '/tmp/myadiyaman/semkitti_ready', 
                                        split = 'validation', 
                                        transform=transform_valid,
                                        iswaymo = False)
    """

    if train_dataset_kitti:
        train_dataset = ConcatDataset([train_dataset_waymo, train_dataset_kitti])
    else:
        train_dataset = train_dataset_waymo

    DEBUG = False
    batch_size = 2 if DEBUG else 12
    workers = 2 if DEBUG else 4

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    #validation_dataloader_kitti = torch.utils.data.DataLoader(validation_dataset_kitti, batch_size=batch_size, 
    #                                                          shuffle=False, drop_last=True, num_workers=workers)
    validation_dataloader_waymo = torch.utils.data.DataLoader(validation_dataset_waymo, batch_size=batch_size, 
                                                              shuffle=False, drop_last=True, num_workers=workers)


    num_classes = 23
    num_epochs = 150
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cw1 = [0.01] + 22*[1]
    criterion1 = nn.CrossEntropyLoss(weight=torch.tensor(cw1).to(device))
    cw2 = [7.11247930e-07, 1.64771221e-04, 1.47421247e-03, 3.49848217e-03,
        6.26231026e-03, 2.27587046e+01, 4.60880581e-02, 1.93580265e-03,
        2.35287205e-03, 2.27660368e-02, 1.32405191e-03, 2.34380348e-02,
        6.66365628e-02, 5.80129043e-02, 4.69185858e-05, 7.16881522e-05,
        8.43246483e-04, 1.01597708e-03, 6.00544630e-05, 2.10271462e-03,
        2.72664635e-03, 1.53651625e-04, 2.49313465e-04]
    criterion2 = nn.CrossEntropyLoss(weight=torch.tensor(cw2).to(device))

    cw3 = [7.1125e-07, 1.6477e-04, 1.4742e-03, 3.4985e-03, 
        6.2623e-03, 4.5906e+00,4.5906e+00, 1.9358e-03, 
        2.3529e-03, 2.2766e-02, 1.3241e-03, 4.5906e+00,
        4.5906e+00, 4.5906e+00, 4.6919e-05, 7.1688e-05, 
        8.4325e-04, 1.0160e-03,6.0054e-05, 2.1027e-03, 
        2.7266e-03, 1.5365e-04, 2.4931e-04]
    criterion3 = nn.CrossEntropyLoss(weight=torch.tensor(cw3).to(device))
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    scheduler = None
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    model.to(device)

    for epoch in range(num_epochs):
        model.to(device)
        model.train()
        running_loss = 0.0
        for i, (images, masks) in tqdm(enumerate(train_dataloader)):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            if i>50 and i%25 < 5:
                loss = criterion1(outputs, masks) + 0.1*criterion2(outputs, masks) + 0.1*criterion3(outputs, masks)
            else:
                loss = criterion1(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_dataloader)}")
        
        if epoch>35 and scheduler:
            scheduler.step()

        # Validation phase
        model.eval()
       
        with torch.no_grad():
            """
            miou_total = 0.0
            running_loss = 0.0
            batch_results = []

            for i, (images, masks) in enumerate(validation_dataloader_kitti):
                images = images.to(torch.float32).to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion1(output, masks)
                running_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)

                # mIoU calculation (excluding class 0)
                miou = calculate_miou(preds, masks, num_classes, ignore_index=0)
                cwiou = calculate_classwise_intersection_union(preds, masks)
                batch_results.append(cwiou)
                
                miou_total += miou
            print(' ')
            print(' ')    
            print('##################KITTI#########################')
            classwise_iou, mean_iou, total_iou = calculate_final_miou_from_batches(batch_results)
            print_miou_results(classwise_iou, mean_iou, total_iou)
            print('################################################')

            avg_miou_kitti = miou_total / len(validation_dataloader_kitti)

            print(f"Epoch [{epoch+1}/{num_epochs}], Kitti Validation Loss: {running_loss / len(validation_dataloader_kitti)}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Kitti Validation mIoU: {avg_miou_kitti:.4f}")
            print('################################################')
            print(' ')
            print(' ')
            """
            # Validation phase
            miou_total = 0.0
            running_loss = 0.0
            batch_results = []

            for i, (images, masks) in enumerate(validation_dataloader_waymo):
                images = images.to(torch.float32).to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion1(outputs, masks)
                running_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)

                # mIoU calculation (excluding class 0)
                miou = calculate_miou(preds, masks, num_classes, ignore_index=0)
                cwiou = calculate_classwise_intersection_union(preds, masks)
                batch_results.append(cwiou)
                
                miou_total += miou
                
            print(' ')
            print(' ')    
            print('##################WAYMO#########################')
            classwise_iou, mean_iou, total_iou = calculate_final_miou_from_batches(batch_results)
            print_miou_results(classwise_iou, mean_iou, total_iou)
            print('################################################')

            avg_miou_waymo = miou_total / len(validation_dataloader_waymo)

            print(f"Epoch [{epoch+1}/{num_epochs}], Waymo Validation Loss: {running_loss / len(validation_dataloader_waymo)}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Waymo Validation mIoU: {avg_miou_waymo:.4f}")
            print('################################################')
            print(' ')
            print(' ')

        if (epoch+1)%20 == 0:
            torch.save(model.cpu().state_dict(), f'midaswaymo_state_dict_{epoch}.pth')

    torch.save(model.cpu().state_dict(), 'midaswaymo_state_dict.pth')


