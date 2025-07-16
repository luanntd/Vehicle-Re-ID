import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OSNet(nn.Module):
    """
    Omni-Scale Network for Vehicle Re-ID
    State-of-the-art architecture for re-identification tasks
    """
    def __init__(self, num_classes=1000, feature_dim=512):
        super(OSNet, self).__init__()
        
        # Using EfficientNet as backbone for better performance
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        
        # Get feature dimension from backbone
        backbone_dim = 1408  # EfficientNet-B2 output dim
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection layers
        self.feature_layers = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        # Classification head for training
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone.features(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Feature projection
        feat = self.feature_layers(pooled)
        
        if self.training:
            # Return both features and classification logits during training
            logits = self.classifier(feat)
            return feat, logits
        else:
            # Return only features during inference
            return feat

class ResNetIBN(nn.Module):
    """
    ResNet with Instance-Batch Normalization
    Better for domain adaptation across different cameras
    """
    def __init__(self, num_classes=1000, feature_dim=2048):
        super(ResNetIBN, self).__init__()
        
        # Load ResNet50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove avgpool and fc layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add IBN layers (simplified version)
        self.ibn_layers = self._make_ibn_layers()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection
        self.feature_layers = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        self.feature_dim = feature_dim
        
    def _make_ibn_layers(self):
        # Simplified IBN implementation
        return nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.InstanceNorm2d(2048, affine=True)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply IBN
        features = self.ibn_layers(features)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Feature projection
        feat = self.feature_layers(pooled)
        
        if self.training:
            logits = self.classifier(feat)
            return feat, logits
        else:
            return feat

class VehicleDescriptor:
    def __init__(self,
                 model_path=None,
                 model_type='osnet',  # 'osnet', 'resnet_ibn', 'efficientnet'
                 input_size=(256, 256),
                 feature_dim=512,
                 num_classes=1000):
        
        self.model_path = model_path
        self.input_size = input_size
        self.model_type = model_type
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        print(f"Using model type: {model_type} with input size: {input_size}, num classes {num_classes} and feature dimension: {feature_dim}")
        # Initialize the model based on type
        if model_type == 'osnet':
            self.model = OSNet(num_classes=num_classes, feature_dim=feature_dim)
        elif model_type == 'resnet_ibn':
            self.model = ResNetIBN(num_classes=num_classes, feature_dim=feature_dim)
        elif model_type == 'efficientnet':
            self.model = self._create_efficientnet_model(feature_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load custom weights if provided
        if self.model_path:
            self._load_model_weights(model_path)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize transform pipeline with better augmentations
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Transforms for training (with augmentation)
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1)), antialias=True),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def _create_efficientnet_model(self, feature_dim):
        """Create EfficientNet-based model for vehicle re-ID"""
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        
        # Replace classifier
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        return model
    
    def _load_model_weights(self, model_path):
        """Load model weights with error handling"""
        try:
            if self.model_type == 'efficientnet':
                # For EfficientNet, load full state dict
                self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            else:
                # For custom models (OSNet, ResNetIBN), load with strict=False
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove classifier layers
                state_dict_filtered = {}
                for k, v in state_dict.items():
                    if k.startswith('classifier.'):
                        print(f"Skipping classifier layer: {k} (shape mismatch)")
                        continue
                    state_dict_filtered[k] = v
                
                self.model.load_state_dict(state_dict_filtered, strict=False)
            print(f"Successfully loaded model weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model weights from {model_path}: {e}")
            print("Using default pretrained weights")

    def extract_feature(self, input_img: np.ndarray, use_tta=False) -> torch.Tensor:
        """
        Extract features from a vehicle image.
        
        Parameters
        ----------
        input_img: np.ndarray
            The vehicle image to extract features from
        use_tta: bool
            Whether to use Test Time Augmentation for better features
            
        Returns
        -------
        torch.Tensor: Feature vector representing the vehicle
        """
        if use_tta:
            return self._extract_feature_with_tta(input_img)
        
        # Preprocess image
        img = self.transforms(input_img).unsqueeze(0)
        img = img.to(device)
        
        # Extract features
        with torch.no_grad():
            if self.model_type == 'efficientnet':
                features = self.model(img)
            else:
                features = self.model(img)
        
        # Handle different output formats
        if isinstance(features, tuple):
            features = features[0]  # Take features, ignore classification logits
            
        features = features.squeeze()
        
        # Normalize features
        features = F.normalize(features, p=2, dim=0)
        
        return features
    
    def _extract_feature_with_tta(self, input_img: np.ndarray) -> torch.Tensor:
        """Extract features using Test Time Augmentation"""
        # Original image
        features_list = []
        
        # Original
        img = self.transforms(input_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = self.model(img)
            if isinstance(feat, tuple):
                feat = feat[0]
            features_list.append(feat.squeeze())
        
        # Horizontal flip
        img_flip = torch.flip(img, dims=[3])
        with torch.no_grad():
            feat_flip = self.model(img_flip)
            if isinstance(feat_flip, tuple):
                feat_flip = feat_flip[0]
            features_list.append(feat_flip.squeeze())
        
        # Average features
        features = torch.stack(features_list).mean(dim=0)
        
        # Normalize
        features = F.normalize(features, p=2, dim=0)
        
        return features

    def get_model_info(self):
        """Get information about the current model"""
        return {
            'model_type': self.model_type,
            'input_size': self.input_size,
            'feature_dim': self.feature_dim,
            'device': str(device),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }

# Example usage and model factory
def create_vehicle_descriptor(model_type='osnet', model_path=None, input_size=(256, 256), num_classes=1000):
    """
    Factory function to create VehicleDescriptor with different models
    
    Available pretrained models:
    - OSNet: Best for vehicle re-ID, handles scale variations well
    - ResNet-IBN: Good for cross-camera scenarios
    - EfficientNet: Efficient and accurate
    """
    return VehicleDescriptor(
        model_path=model_path,
        model_type=model_type,
        input_size=input_size,
        num_classes=num_classes
    )