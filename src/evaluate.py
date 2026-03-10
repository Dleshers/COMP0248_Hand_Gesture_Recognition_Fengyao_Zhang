import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import HandGestureDataset
#from model import HandGestureNet #for rgb only
from model import RGBD_TwoStreamNet #two-stream

def compute_bbox_iou(box1, box2):
    # box format: [x_min, y_min, x_max, y_max]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - iter_area

    if union_area == 0:
        return 0.0
    else :
        return (iter_area / union_area).item()
    
def compute_mask_metrics(pred_mask, true_mask):
    pred = pred_mask.bool()
    true = true_mask.bool()

    intersection = (pred & true).float().sum().item()

    # Union is the total area covered by both masks
    union = (pred | true).float().sum().item()

    if union > 0:
        iou = intersection / union
    else:
        iou = 0.0

    # Dice coefficient calculation
    dice = (2. * intersection) / (pred.float().sum().item() + true.float().sum().item() + 1e-6)

    return iou, dice
    
def evaluate(model_path, data_dir='dataset/val_depth', batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = HandGestureDataset(root_dir=data_dir, use_depth=True, is_train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # model = HandGestureNet(in_channels=3, num_classes=10).to(device) #rgb only
    model = RGBD_TwoStreamNet(num_classes=10).to(device) #two-stream
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Metrics accumulators
    all_preds_cls = []
    all_trues_cls = []
    
    iou_bboxes = []
    acc_05_bboxes = []
    
    iou_masks = []
    dice_masks = []

    print(f"\n Starting evaluation on {data_dir}...")

    with torch.no_grad():
        for images_rgb, images_depth, targets in tqdm(loader, desc="Evaluating", unit="batch"):
            #images = images.to(device) #rgb only
            images_rgb = images_rgb.to(device)
            images_depth = images_depth.to(device) #two-stream  
            labels = targets['labels'].squeeze(-1).to(device)
            masks = targets['masks'].float().to(device)
            boxes = targets['boxes'].squeeze(1).to(device)

            #h, w = images.shape[2], images.shape[3] #rgb only
            h, w = images_rgb.shape[2], images_rgb.shape[3] #two-stream
            boxes_norm = boxes.clone()
            boxes_norm[:, 0] /= w
            boxes_norm[:, 1] /= h
            boxes_norm[:, 2] /= w
            boxes_norm[:, 3] /= h

            #outputs = model(images) #rgb only
            outputs = model(images_rgb, images_depth) #two-stream

            # Classification predictions
            preds_cls = torch.argmax(outputs['cls_logits'], dim=1)
            all_preds_cls.extend(preds_cls.cpu().numpy())
            all_trues_cls.extend(labels.cpu().numpy())

            #Bounding box metrics
            preds_bbox = outputs['bbox_norm']
            for i in range(len(preds_bbox)):
                iou = compute_bbox_iou(preds_bbox[i], boxes_norm[i])
                iou_bboxes.append(iou)

                if iou >= 0.5:
                    acc_05_bboxes.append(1)
                else:
                    acc_05_bboxes.append(0)

            #Segmentation metrics
            preds_mask = torch.sigmoid(outputs['mask_logits']) > 0.5
            for i in range(len(preds_mask)):
                iou_m, dice_m = compute_mask_metrics(preds_mask[i].squeeze(0), masks[i].squeeze(0))
                iou_masks.append(iou_m)
                dice_masks.append(dice_m)

    # Final metrics calculation
    # Detection metrics
    mean_bbox_iou = np.mean(iou_bboxes)
    det_acc_05 = np.mean(acc_05_bboxes)

    #Segmentation metrics
    mean_mask_iou = np.mean(iou_masks)
    mean_dice = np.mean(dice_masks)

    #Classification metrics
    acc_top1 = np.mean(np.array(all_preds_cls) == np.array(all_trues_cls))
    f1_macro = f1_score(all_trues_cls, all_preds_cls, average='macro')
    conf_matrix = confusion_matrix(all_trues_cls, all_preds_cls)

    #print results
    print(f"Detection - Mean BBox IoU: {mean_bbox_iou:.4f}")
    print(f"Detection - Accuracy IoU: {det_acc_05:.4f}")
    print(f"Segmentation - Mean Mask IoU: {mean_mask_iou:.4f}")
    print(f"Segmentation - Mean Dice Coeff: {mean_dice:.4f}")
    print(f"Classification - Top-1 Accuracy: {acc_top1:.4f}")
    print(f"Classification - Macro F1 Score: {f1_macro:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

if __name__ == '__main__':
    # evaluate('weights/hand_gesture_epoch_100.pth', data_dir='dataset/test') #rgb only
    evaluate('weights/rgbd_two_stream_attention_new_epoch_100.pth', data_dir='dataset/val_depth') #two-stream

    

