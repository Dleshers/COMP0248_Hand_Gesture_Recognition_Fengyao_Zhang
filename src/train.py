import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#from model import HandGestureNet #only rgb
from model import RGBD_TwoStreamNet #two-stream
from dataloader import HandGestureDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters:
    batch_size = 8
    epochs = 100
    lr = 1e-4

    # Get the absolute path to the dataset
    current_src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_src_dir)
    train_data_dir = os.path.join(project_root_dir, 'dataset', 'train_depth') #two-stream
    
    print(f"Data dir: {train_data_dir}")

    # Initialize dataset and dataloader
    train_dataset = HandGestureDataset(root_dir=train_data_dir, use_depth=True)
    
    print(f"The number of samples in the dataset is: {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise RuntimeError("dataset is empty.")

    # set num_workers=0:on Windows only, otherwise it may cause issues with multiprocessing.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Initialize model, loss functions, and optimizer
    # model = HandGestureNet(in_channels=3, num_classes=10).to(device) #rgb only
    model = RGBD_TwoStreamNet(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision training setup 
    scaler = torch.amp.GradScaler('cuda')

    # Loss functions
    cri_cls = nn.CrossEntropyLoss() # For classification
    cri_seg = nn.BCEWithLogitsLoss() # For segmentation (binary mask)
    cri_det = nn.MSELoss() # For bounding box regression

    #Loss weights
    w_cls = 3.0
    w_seg = 0.5
    w_det = 0.5

    # create the folder for weights
    os.makedirs('weights', exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # for batch_idx, (images, targets) in enumerate(train_loader): #rgb only
        for batch_idx, (images_rgb, images_depth, targets) in enumerate(train_loader):
            #images = images.to(device) #rgb only
            images_rgb = images_rgb.to(device)
            images_depth = images_depth.to(device)

            #Format targets
            labels = targets['labels'].squeeze(-1).to(device) 
            masks = targets['masks'].float().to(device)
            boxes = targets['boxes'].squeeze(1).to(device)

            # Normalize bounding boxes to [0, 1] range
            h = images_rgb.shape[2]
            w = images_rgb.shape[3]
            boxes_norm = boxes.clone()
            boxes_norm[:, 0] /= w
            boxes_norm[:, 1] /= h
            boxes_norm[:, 2] /= w
            boxes_norm[:, 3] /= h

            optimizer.zero_grad()

            
            #Forward pass
            with torch.autocast(device_type='cuda'):
                # Forward pass (two-stream)
                # outputs = model(images) #rgb only
                outputs = model(images_rgb, images_depth) 

                # compute losses
                loss_cls = cri_cls(outputs['cls_logits'], labels)
                loss_seg = cri_seg(outputs['mask_logits'], masks)
                loss_det = cri_det(outputs['bbox_norm'], boxes_norm)

                # Combined Loss
                loss = w_cls * loss_cls + w_seg * loss_seg + w_det * loss_det

            #Backward and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Print training progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f} (Cls: {loss_cls.item():.4f}, "
                      f"Seg: {loss_seg.item():.4f}, Det: {loss_det.item():.4f})")
                
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

        # save weights
        save_interval = 10 
        
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            # save_path = f'weights/hand_gesture_epoch_{epoch+1}.pth' #rgb only
            # save_path = f'weights/rgbd_two_stream_epoch_{epoch+1}.pth' #two-stream
            save_path = os.path.join('weights', f'rgbd_two_stream_attention_new_epoch_{epoch+1}.pth') #two-stream with attention
            torch.save(model.state_dict(), save_path)

        scheduler.step()

if __name__ == '__main__':
    #print("test:main")
    train()

    