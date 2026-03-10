import torch
import torch.nn as nn
import torch.nn.functional as F

class StreamEncoder(nn.Module):
    #Independent feature extraction stream
    def __init__(self, in_channels):
        super().__init__()
        
        self.encoder1 = self._conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4) 
        return p4

class ChannelAttention(nn.Module):
    #Squeeze-and-Excitation Attention Block for Modality Reweighting

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Two fully connected layers to learn channel weights
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid() # Outputs weights between 0 and 1
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Learn channel weights
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # Reweight the original feature map dynamically
        return x * y.expand_as(x)


class RGBD_TwoStreamNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        

        # Parallel Modality Streams

        self.rgb_stream = StreamEncoder(in_channels=3)
        self.depth_stream = StreamEncoder(in_channels=1)

        # 512 RGB + 512 Depth channels will be filtered here
        self.attention = ChannelAttention(in_channels=1024)
        

        # Late Fusion Layer 
        # seamlessly connecting to the original multi-task heads.
        self.fusion_conv = nn.Conv2d(1024, 512, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(512)
        

        # Segmentation Head
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(256, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 128)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(64, 64)
        
        self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.seg_out = nn.Conv2d(64, 1, kernel_size=1)

        # Global Feature Pooling

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


        # Classification & Detection Heads

        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.det_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid() 
        )

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_rgb, x_depth):
        
        # Feature extraction through respective modality streams
        feat_rgb = self.rgb_stream(x_rgb)       
        feat_depth = self.depth_stream(x_depth) 
        
        # Concatenate along the channel dimension
        fused = torch.cat((feat_rgb, feat_depth), dim=1) 

        # Dynamically reweighting the concatenated features to emphasize more informative channels
        fused = self.attention(fused)
        
        # Dimensionality Reduction: 1x1 convolution compresses fused features back to 512 channels
        p4 = F.relu(self.fusion_bn(self.fusion_conv(fused))) 
        
        # Segmentation Path 
        d1 = self.upconv1(p4)
        d1 = self.dec1(d1)
        d2 = self.upconv2(d1)
        d2 = self.dec2(d2)
        d3 = self.upconv3(d2)
        d3 = self.dec3(d3)
        d4 = self.upconv4(d3)
        mask_logits = self.seg_out(d4) 

        # Classification and Detection Path
        pooled = self.global_pool(p4)
        pooled = torch.flatten(pooled, 1) 

        cls_logits = self.cls_head(pooled)
        bbox_norm = self.det_head(pooled)

        return {
            "mask_logits": mask_logits,
            "cls_logits": cls_logits,
            "bbox_norm": bbox_norm
        }

# Testing the model
if __name__ == "__main__":
    # Simulate a batch of 2 RGB images and 2 Depth images
    dummy_rgb = torch.randn(2, 3, 256, 256)
    dummy_depth = torch.randn(2, 1, 256, 256)
    
    # Initialize model
    model = RGBD_TwoStreamNet(num_classes=10)
    
    # Forward pass
    outputs = model(dummy_rgb, dummy_depth)
    

    print(f"Mask Logits: {outputs['mask_logits'].shape} ") 
    print(f"Class Logits: {outputs['cls_logits'].shape} ") 
    print(f"BBox (Norm): {outputs['bbox_norm'].shape} ")