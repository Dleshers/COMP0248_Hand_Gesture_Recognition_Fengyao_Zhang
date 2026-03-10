import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from dataloader import HandGestureDataset
from model import RGBD_TwoStreamNet #two-stream
#from model import HandGestureNet #for rgb only

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # Unnormalize a tensor image for visualization
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img_np = tensor.cpu().numpy()
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0.0, 1.0) # 防止高光过曝产生噪点
    return np.transpose(img_np, (1, 2, 0)) # 转换为 (H, W, C) 给 matplotlib


def visualize_predictions(model_path, data_dir='dataset/test', output_dir='results', num_samples=5):
    # Set up device and load model
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running visualization on device: {device}")

    # Initialize dataset and dataloader
    # dataset = HandGestureDataset(root_dir=data_dir, use_depth=False) #rgb only
    dataset = HandGestureDataset(root_dir=data_dir, use_depth=True) #two-stream
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # mapping from class index to gesture name
    idx_to_class = {v: k.split('_')[1] for k, v in dataset.class_to_idx.items()}

    # Load model
    model = RGBD_TwoStreamNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    count=0
    with torch.no_grad():
        #for images, targets in loader: #rgb only
        for images_rgb, images_depth, targets in loader: #two-stream
            if count >= num_samples:
                break
            
            #images = images.to(device) #rgb only

            images_rgb = images_rgb.to(device)
            images_depth = images_depth.to(device) #two-stream
            gt_labels = targets['labels'].item()
            gt_boxes = targets['boxes'].squeeze(0).squeeze(0).cpu().numpy()

            # model inference
            #outputs = model(images) #rgb only
            outputs = model(images_rgb, images_depth) #two-stream
           

            # predictions
            pred_cls_idx = torch.argmax(outputs['cls_logits'], dim=1).item()
            pred_cls_name = idx_to_class[pred_cls_idx]
            gt_cls_name = idx_to_class[gt_labels]

            # Apply sigmoid and threshold for segmentation mask
            pred_mask = (torch.sigmoid(outputs['mask_logits']) > 0.5).squeeze().cpu().numpy()
            
            # Denormalize bounding box coordinates back to image scale
            #h, w = images.shape[2], images.shape[3] #rgb only
            h, w = images_rgb.shape[2], images_rgb.shape[3] #two-stream
            pred_box_norm = outputs['bbox_norm'].squeeze().cpu().numpy()
            pred_box = pred_box_norm * np.array([w, h, w, h])
            
            img_np = unnormalize(images_rgb.squeeze(0))

            # Plotting
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(img_np)
            
            # Overlay predicted mask 
            mask_colored = np.zeros((*pred_mask.shape, 4))
            mask_colored[pred_mask == 1] = [0, 1, 0, 0.5] 
            ax.imshow(mask_colored)
            
            # Draw predicted bounding box
            xmin, ymin, xmax, ymax = pred_box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     linewidth=2, edgecolor='r', facecolor='none', label='Pred Box')
            ax.add_patch(rect)
            
            # Add text annotation
            title_color = 'green' if pred_cls_idx == gt_labels else 'red'
            ax.set_title(f"Pred: {pred_cls_name} | GT: {gt_cls_name}", color=title_color, fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Save the figure
            #save_path = os.path.join(output_dir, f'vis_sample_test_{count+1}.png') #rgb only
            save_path = os.path.join(output_dir, f'vis_sample_rgbd_test_new_{count+1}.png') #two-stream
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            print(f"Saved visualization to {save_path}")
            count += 1

if __name__ == '__main__':
    
    # get the current directories and test data path 
    current_src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_src_dir)
    
    test_data_dir = os.path.join(project_root_dir, 'dataset', 'test')
    output_results_dir = os.path.join(project_root_dir, 'results')
    # weights_path = os.path.join(project_root_dir, 'weights', 'hand_gesture_epoch_100.pth') #rgb only
    weights_path = os.path.join(project_root_dir, 'weights', 'rgbd_two_stream_attention_new_epoch_100.pth') #two-stream
    
    print(f"try to load data: {test_data_dir}")
    print(f"try to load weights: {weights_path}")
    
    visualize_predictions(
        model_path=weights_path, 
        data_dir=test_data_dir, 
        output_dir=output_results_dir,
        num_samples=150  # visualize samples for demonstration:must be changed for differnt datasets
    )


