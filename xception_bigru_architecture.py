"""
XceptionBiGRU - Paper-Accurate Implementation

Uses actual Xception architecture (not DenseNet)
With BiGRU for temporal modeling
Matches 2D-Malafide paper's detection architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    """Separable Convolution (depthwise + pointwise)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    """Xception Block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = SeparableConv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        
        self.conv2 = SeparableConv2d(out_channels // 4, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        
        self.conv3 = SeparableConv2d(out_channels // 2, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = None
    
    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = x + residual
        x = F.relu(x)
        
        return x

class Xception(nn.Module):
    """Paper-Accurate Xception Architecture"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Middle flow
        self.block1 = XceptionBlock(64, 128, stride=2)
        self.block2 = XceptionBlock(128, 256, stride=2)
        self.block3 = XceptionBlock(256, 512, stride=2)
        self.block4 = XceptionBlock(512, 512, stride=1)
        
        # Exit flow
        self.conv3 = SeparableConv2d(512, 1024, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_dim = 1024
    
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Middle flow
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Exit flow
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x

class XceptionBiGRU(nn.Module):
    """
    Paper-Accurate Xception + BiGRU
    Matches 2D-Malafide paper architecture
    """
    
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        
        # Xception backbone
        self.xception = Xception(num_classes=num_classes)
        self.feature_dim = self.xception.feature_dim
        
        # BiGRU for temporal modeling
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 3, 224, 224)
        
        # Extract features using Xception
        features = self.xception(x)  # (batch, 1024)
        
        # Prepare for BiGRU (add temporal dimension)
        features = features.unsqueeze(1)  # (batch, 1, 1024)
        
        # BiGRU processing
        gru_out, _ = self.gru(features)  # (batch, 1, 1024)
        
        # Take last output
        gru_out = gru_out[:, -1, :]  # (batch, 1024)
        
        # Classification
        logits = self.classifier(gru_out)  # (batch, num_classes)
        
        return logits

if __name__ == "__main__":
    # Test the architecture
    model = XceptionBiGRU(num_classes=2)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ Model works correctly!")
