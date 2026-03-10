import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFilter
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

class HandGestureDataset(Dataset):
    def __init__(self, root_dir, use_depth=True, transforms=None,is_train=True):

        self.root_dir = root_dir
        self.use_depth = use_depth
        self.transforms = transforms
        self.is_train = is_train # the flag of enforcement of training
        
        self.output_size = 224               # final resize size 
        self.expand_ratio = 1.3              # expand the bounding box by this ratio
        self.rotation_deg = 15               # ±rotation_deg
        self.flip_prob = 0.5                 # probability of horizontal flip
        self.perspective_prob = 0.2          # probability of perspective transform
        self.blur_prob = 0.15                # probability of background blur
        self.depth_noise_std = 0.02          # standard deviation of depth noise
        # RGB normalization 
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        # depth normalization 
        self.depth_mean = 0.4
        self.depth_std = 0.25



        # Define a mapping dictionary for the 10 predefined classes
        self.class_to_idx = {
            'G01_call': 0, 'G02_dislike': 1, 'G03_like': 2, 'G04_ok': 3,
            'G05_one': 4, 'G06_palm': 5, 'G07_peace': 6, 'G08_rock': 7,
            'G09_stop': 8, 'G10_three': 9
        }
        
        self.samples = []
        self._load_samples()

        # A flag to print depth data
        self.has_printed_debug = False

        # Define a color jitter transform for data augmentation during training
        self.color_jitter = T.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.05
        )

        # Define a random erasing transform for depth data augmentation during training
        self.depth_eraser = T.RandomErasing(
            p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.0
        )



    def _load_samples(self):

        if not os.path.exists(self.root_dir):
            print(f"Warning: Dataset root directory {self.root_dir} does not exist.")
            return

        # Iterate throgh every folders
        for student_folder in os.listdir(self.root_dir):
            student_path = os.path.join(self.root_dir, student_folder)
            if not os.path.isdir(student_path):
                continue
                
            for gesture, label_idx in self.class_to_idx.items():
                gesture_path = os.path.join(student_path, gesture)
                if not os.path.isdir(gesture_path):
                    continue
                    
                for clip in os.listdir(gesture_path):
                    clip_path = os.path.join(gesture_path, clip)
                    if not os.path.isdir(clip_path):
                        continue
                        
                    annotation_dir = os.path.join(clip_path, 'annotation')
                    if not os.path.exists(annotation_dir):
                        continue
                        
                    for mask_name in os.listdir(annotation_dir):
                        if not mask_name.endswith('.png'):
                            continue
                            
                        mask_path = os.path.join(annotation_dir, mask_name)
                        rgb_path = os.path.join(clip_path, 'rgb', mask_name)
                        depth_path = os.path.join(clip_path, 'depth_raw', mask_name.replace('.png', '.npy')) 

                        # Basic path existence check
                        if os.path.exists(rgb_path) and os.path.exists(mask_path):
                            self.samples.append({
                                'rgb': rgb_path,
                                'depth': depth_path if self.use_depth else None,
                                'mask': mask_path,
                                'label': label_idx
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            sample = self.samples[idx]
            
            try:
                # Basic Image Loading
                image = Image.open(sample['rgb']).convert("RGB")
                mask_img = Image.open(sample['mask']).convert("L")
                
                # Depth Data Loading & Absolute Physical Clipping
                if self.use_depth and sample['depth']:
                    if not os.path.exists(sample['depth']):
                        raise FileNotFoundError(f"Missing depth file: {sample['depth']}")
                    
                    depth_array = np.load(sample['depth']).astype(np.float32)
                    depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)

                    # Extract valid depth mask
                    valid_mask = depth_array > 0.0
                    
                    # Absolute Physical Clipping: Focus strictly on the 200mm to 1500mm interactive space
                    depth_array = np.clip(depth_array, a_min=200.0, a_max=1500.0)
                    
                    # Restore dead pixels to 0.0 to prevent them from being incorrectly normalized as 200mm
                    depth_array[~valid_mask] = 0.0
                    
                    # Fixed Boundary Normalization: Ensures the network's absolute distance perception 
                    depth_array[valid_mask] = (depth_array[valid_mask] - 200.0) / (1500.0 - 200.0)
                    
                    depth_tensor = torch.from_numpy(depth_array).unsqueeze(0) 

                    if not self.has_printed_debug:
                        print("Depth Data Successfully Loaded & Physi-Clipped!")
                        self.has_printed_debug = True


                # Compute bbox from original mask (before any augmentation)
                mask_np_orig = np.array(mask_img)
                mask_bin = mask_np_orig > 127
                pos = np.where(mask_bin)
                if len(pos[0]) > 0:
                    xmin = int(np.min(pos[1]))
                    xmax = int(np.max(pos[1]))
                    ymin = int(np.min(pos[0]))
                    ymax = int(np.max(pos[0]))

                    # expand bbox by expand_ratio (around center)
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    bw = (xmax - xmin) * self.expand_ratio
                    bh = (ymax - ymin) * self.expand_ratio

                    left = int(max(cx - bw / 2.0, 0))
                    top = int(max(cy - bh / 2.0, 0))
                    right = int(min(cx + bw / 2.0, image.width))
                    bottom = int(min(cy + bh / 2.0, image.height))

                    crop_left = left
                    crop_top = top
                    crop_w = max(1, right - left)
                    crop_h = max(1, bottom - top)
                else:
                    # fallback: use full image
                    crop_left = 0
                    crop_top = 0
                    crop_w = image.width
                    crop_h = image.height

                # Crop RGB / mask / depth synchronously
                image = TF.crop(image, crop_top, crop_left, crop_h, crop_w)
                mask_img = TF.crop(mask_img, crop_top, crop_left, crop_h, crop_w)
                if depth_tensor is not None:
                    depth_tensor = TF.crop(depth_tensor, crop_top, crop_left, crop_h, crop_w)

                # Core Data Augmentation Pipeline 
                if self.is_train:
                    # Random horizontal flip 
                    if random.random() < self.flip_prob:
                        image = TF.hflip(image)
                        mask_img = TF.hflip(mask_img)
                        if depth_tensor is not None:
                            depth_tensor = TF.hflip(depth_tensor)

                    # Random rotation ±rotation_deg 
                    angle = random.uniform(-self.rotation_deg, self.rotation_deg)
                    image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                    mask_img = TF.rotate(mask_img, angle, interpolation=InterpolationMode.NEAREST)
                    if depth_tensor is not None:
                        # for depth, use bilinear-like interpolation if supported
                        depth_tensor = TF.rotate(depth_tensor, angle, interpolation=InterpolationMode.BILINEAR)

                    # Random small perspective 
                    if random.random() < self.perspective_prob:
                        # use torchvision to get params; implement a simple small perspective transform
                        # For simplicity, we create small random shifts of corners
                        w, h = image.size
                        shift = 0.02  # proportion
                        def jitter_point(x, y):
                            return (x + random.uniform(-shift, shift) * w, y + random.uniform(-shift, shift) * h)
                        startpoints = [(0, 0), (w, 0), (w, h), (0, h)]
                        endpoints = [jitter_point(x, y) for (x, y) in startpoints]
                        image = TF.perspective(image, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR)
                        mask_img = TF.perspective(mask_img, startpoints, endpoints, interpolation=InterpolationMode.NEAREST)
                        if depth_tensor is not None:
                            depth_tensor = TF.perspective(depth_tensor, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR)

                    # RGB-specific: Color and Brightness Jitter
                    if random.random() > 0.5:
                        image = self.color_jitter(image)

                    # Random background blur
                    if random.random() < self.blur_prob:
                        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.8)))

                    # Depth-specific: add gaussian noise + Random Erasing
                    if depth_tensor is not None:
                        if random.random() < 0.5 and self.depth_noise_std > 0:
                            noise = torch.randn_like(depth_tensor) * self.depth_noise_std
                            depth_tensor = depth_tensor + noise
                            depth_tensor = torch.clamp(depth_tensor, 0.0, 1.0)
                        if random.random() < 0.1:
                            # Apply random erasing to depth tensor (value 0 indicates missing)
                            depth_tensor = self.depth_eraser(depth_tensor)
                

               
                # Resize image & mask & depth to final size 
                image = TF.resize(image, [self.output_size, self.output_size], interpolation=InterpolationMode.BILINEAR)
                mask_img = TF.resize(mask_img, [self.output_size, self.output_size], interpolation=InterpolationMode.NEAREST)
                if depth_tensor is not None:
                    depth_tensor = TF.resize(depth_tensor, [self.output_size, self.output_size])

                # Re-binarize mask after all transforms
                mask = np.array(mask_img)
                mask = mask > 127

                obj_ids = np.unique(mask)
                if len(obj_ids) <= 1:
                    # If no object is present after augmentation, we can either skip this sample or return a dummy box/mask. Here we choose to skip.
                    raise ValueError(f"Mask is all black after augmentation: {sample['mask']}")

                pos = np.where(mask)
                if len(pos[0]) > 0:
                    xmin = int(np.min(pos[1]))
                    xmax = int(np.max(pos[1]))
                    ymin = int(np.min(pos[0]))
                    ymax = int(np.max(pos[0]))

                    # Validate bounding box coordinates
                    if xmax <= xmin or ymax <= ymin:
                        raise ValueError(f"Extracted bounding box is invalid after augmentation: {sample['mask']}")

                    boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
                else:
                    raise ValueError(f"Failed to extract bounding box after augmentation: {sample['mask']}")
                

                masks = torch.as_tensor(mask.astype(np.uint8), dtype=torch.uint8).unsqueeze(0)
                labels = torch.as_tensor([sample['label']], dtype=torch.int64)

                target = {
                    "boxes": boxes,
                    "labels": labels,
                    "masks": masks
                }

                # Apply External Transforms
                if self.transforms is not None:
                    image, target = self.transforms(image, target)
                else:
                    # Convert image to tensor and normalize
                    image = TF.to_tensor(image)
                    image = TF.normalize(image, mean=self.rgb_mean, std=self.rgb_std)

                # Depth final normalization (
                if self.use_depth and depth_tensor is not None:
                    # ensure depth_tensor is float tensor shape
                    if not isinstance(depth_tensor, torch.Tensor):
                        depth_tensor = torch.from_numpy(np.array(depth_tensor)).unsqueeze(0).float()
                    depth_tensor = depth_tensor.float()
                    # depth already in [0,1] for valid pixels from earlier phys clipping; normalize
                    depth_tensor = (depth_tensor - self.depth_mean) / self.depth_std

                if self.use_depth:
                    return image, depth_tensor, target
                else:
                    return image, target

            except Exception as e:
                # Catch any reading or processing errors, silently skip, and randomly select another index to retry
                print(f"Skipping corrupted sample: {e}")
                idx = np.random.randint(0, len(self.samples))