import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import argparse
from pathlib import Path

# Import your enhanced feature extraction models
from realtime_reid.feature_extraction import OSNet, ResNetIBN, create_vehicle_descriptor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VehicleReIDDataset(Dataset):
    """
    Dataset class for Vehicle Re-ID training
    
    Expected data format:
    dataset/
    ├── images/
    │   ├── 0001_c001_001.jpg  # Format: vehicleID_cameraID_frameID.jpg
    │   ├── 0001_c002_001.jpg
    │   ├── 0002_c001_001.jpg
    │   └── ...
    ├── train.txt              # List of training image names
    ├── test.txt               # List of test image names
    └── query.txt              # List of query image names
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'images'
        self.split = split
        self.transform = transform
        
        # Load image list
        split_file = self.data_dir / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]
        
        # Extract vehicle IDs and camera IDs
        self.vehicle_ids = []
        self.camera_ids = []
        
        for img_name in self.image_list:
            # Parse format: vehicleID_cameraID_frameID.jpg
            parts = img_name.split('_')
            vehicle_id = int(parts[0])
            camera_id = int(parts[1][1:])  # Remove 'c' prefix
            
            self.vehicle_ids.append(vehicle_id)
            self.camera_ids.append(camera_id)
        
        # Create label mapping
        unique_ids = sorted(set(self.vehicle_ids))
        self.id_to_label = {vid: idx for idx, vid in enumerate(unique_ids)}
        self.labels = [self.id_to_label[vid] for vid in self.vehicle_ids]
        
        self.num_classes = len(unique_ids)
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        vehicle_id = self.vehicle_ids[idx]
        camera_id = self.camera_ids[idx]
        
        return {
            'image': image,
            'label': label,
            'vehicle_id': vehicle_id,
            'camera_id': camera_id,
            'img_name': img_name
        }

class TripletLoss(nn.Module):
    """Triplet Loss for Re-ID training"""
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, features, labels):
        """
        Args:
            features: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size)
        """
        n = features.size(0)
        
        # Compute pairwise distance matrix
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(features, features.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        
        # For each anchor, find the hardest positive and negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        for i in range(n):
            # Hardest positive
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            # Hardest negative
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss

class CenterLoss(nn.Module):
    """Center Loss for Re-ID training"""
    
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size)
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss

def train_vehicle_reid_model(
    data_dir,
    model_type='osnet',
    num_epochs=120,
    batch_size=32,
    learning_rate=0.0003,
    weight_decay=5e-4,
    step_size=40,
    gamma=0.1,
    margin=0.3,
    save_dir='checkpoints'
):
    """
    Train vehicle re-identification model
    
    Parameters:
    -----------
    data_dir: str
        Path to dataset directory
    model_type: str
        Type of model to train ('osnet', 'resnet_ibn', 'efficientnet')
    num_epochs: int
        Number of training epochs
    batch_size: int
        Training batch size
    learning_rate: float
        Initial learning rate
    weight_decay: float
        Weight decay for optimizer
    step_size: int
        Step size for learning rate scheduler
    gamma: float
        Gamma for learning rate scheduler
    margin: float
        Margin for triplet loss
    save_dir: str
        Directory to save checkpoints
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = VehicleReIDDataset(data_dir, split='train', transform=train_transform)
    val_dataset = VehicleReIDDataset(data_dir, split='test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of identities: {train_dataset.num_classes}")
    
    # Create model
    if model_type == 'osnet':
        model = OSNet(num_classes=train_dataset.num_classes, feature_dim=512)
    elif model_type == 'resnet_ibn':
        model = ResNetIBN(num_classes=train_dataset.num_classes, feature_dim=2048)
    elif model_type == 'efficientnet':
        descriptor = create_vehicle_descriptor('efficientnet')
        model = descriptor.model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Define losses
    criterion_ce = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=margin)
    criterion_center = CenterLoss(train_dataset.num_classes, model.feature_dim, use_gpu=True)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Training loop
    best_mAP = 0.0
    train_losses = []
    val_mAPs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_triplet_loss = 0.0
        running_center_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            # Forward pass
            if model_type in ['osnet', 'resnet_ibn']:
                features, logits = model(images)
            else:  # efficientnet
                features = model(images)
                logits = features  # For EfficientNet, features are the final output
            
            # Compute losses
            ce_loss = criterion_ce(logits, labels)
            triplet_loss = criterion_triplet(features, labels)
            center_loss = criterion_center(features, labels)
            
            # Total loss
            loss = ce_loss + triplet_loss + 0.0005 * center_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update center loss
            for param in criterion_center.parameters():
                param.grad.data *= (1. / 0.0005)
            optimizer_center.step()
            
            # Update running losses
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_triplet_loss += triplet_loss.item()
            running_center_loss += center_loss.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Triplet': f'{triplet_loss.item():.4f}',
                'Center': f'{center_loss.item():.4f}'
            })
        
        # Calculate average training losses
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            mAP = evaluate_model(model, val_loader, model_type)
            val_mAPs.append(mAP)
            
            print(f'Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val mAP = {mAP:.4f}')
            
            # Save best model
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mAP': best_mAP,
                    'model_type': model_type
                }, os.path.join(save_dir, f'best_{model_type}_model.pth'))
                print(f'New best model saved with mAP: {best_mAP:.4f}')
        
        scheduler.step()
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': mAP if (epoch + 1) % 5 == 0 else 0.0,
                'model_type': model_type
            }, os.path.join(save_dir, f'{model_type}_epoch_{epoch+1}.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if val_mAPs:
        plt.subplot(1, 2, 2)
        plt.plot(range(4, len(val_mAPs)*5, 5), val_mAPs)
        plt.title('Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_type}_training_curves.png'))
    plt.show()
    
    print(f'Training completed. Best mAP: {best_mAP:.4f}')

def evaluate_model(model, data_loader, model_type):
    """Evaluate model on validation set"""
    model.eval()
    
    all_features = []
    all_labels = []
    all_camera_ids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Extracting features'):
            images = batch['image'].to(device)
            labels = batch['label']
            camera_ids = batch['camera_id']
            
            if model_type in ['osnet', 'resnet_ibn']:
                features, _ = model(images)
            else:
                features = model(images)
            
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu())
            all_labels.extend(labels.numpy())
            all_camera_ids.extend(camera_ids.numpy())
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)
    all_camera_ids = np.array(all_camera_ids)
    
    # Compute mAP
    mAP = compute_mAP(all_features, all_labels, all_camera_ids)
    
    return mAP

def compute_mAP(features, labels, camera_ids):
    """Compute mean Average Precision for re-identification"""
    n = features.size(0)
    
    # Compute distance matrix
    dist_matrix = torch.cdist(features, features, p=2)
    
    aps = []
    for i in range(n):
        # Get query
        query_label = labels[i]
        query_cam = camera_ids[i]
        
        # Get distances to all other samples
        distances = dist_matrix[i]
        
        # Create mask for valid gallery samples (different camera, exclude query)
        valid_mask = (camera_ids != query_cam) & (np.arange(n) != i)
        
        if not valid_mask.any():
            continue
        
        # Get labels and distances for valid gallery
        gallery_labels = labels[valid_mask]
        gallery_distances = distances[valid_mask]
        
        # Sort by distance
        sorted_indices = torch.argsort(gallery_distances)
        sorted_labels = gallery_labels[sorted_indices]
        
        # Compute precision and recall
        matches = (sorted_labels == query_label)
        
        if not matches.any():
            continue
        
        # Compute average precision
        precision_at_k = []
        num_matches = 0
        
        for k in range(len(matches)):
            if matches[k]:
                num_matches += 1
                precision = num_matches / (k + 1)
                precision_at_k.append(precision)
        
        if precision_at_k:
            ap = np.mean(precision_at_k)
            aps.append(ap)
    
    if aps:
        mAP = np.mean(aps)
    else:
        mAP = 0.0
    
    return mAP

def create_dataset_splits(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Create train/val/test splits from vehicle images
    
    Expected image naming format: vehicleID_cameraID_frameID.jpg
    Example: 0001_c001_001.jpg, 0001_c002_001.jpg, 0002_c001_001.jpg
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / 'images'
    
    # Get all image files
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    # Extract vehicle IDs
    vehicle_dict = {}
    for img_file in image_files:
        img_name = img_file.name
        vehicle_id = img_name.split('_')[0]
        
        if vehicle_id not in vehicle_dict:
            vehicle_dict[vehicle_id] = []
        vehicle_dict[vehicle_id].append(img_name)
    
    # Split vehicle IDs
    vehicle_ids = list(vehicle_dict.keys())
    np.random.shuffle(vehicle_ids)
    
    n_total = len(vehicle_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_ids = vehicle_ids[:n_train]
    val_ids = vehicle_ids[n_train:n_train + n_val]
    test_ids = vehicle_ids[n_train + n_val:]
    
    # Create image lists
    train_images = []
    val_images = []
    test_images = []
    
    for vid in train_ids:
        train_images.extend(vehicle_dict[vid])
    
    for vid in val_ids:
        val_images.extend(vehicle_dict[vid])
    
    for vid in test_ids:
        test_images.extend(vehicle_dict[vid])
    
    # Save splits
    with open(data_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_images))
    
    with open(data_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_images))
    
    with open(data_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(test_images))
    
    # Create query.txt (subset of test for evaluation)
    # Select one image per vehicle ID and camera combination for query
    query_images = []
    for vid in test_ids[:len(test_ids)//2]:  # Use half of test IDs for query
        vid_images = vehicle_dict[vid]
        if vid_images:
            query_images.append(vid_images[0])  # Take first image
    
    with open(data_dir / 'query.txt', 'w') as f:
        f.write('\n'.join(query_images))
    
    print(f"Dataset splits created:")
    print(f"Train: {len(train_images)} images from {len(train_ids)} vehicles")
    print(f"Val: {len(val_images)} images from {len(val_ids)} vehicles")
    print(f"Test: {len(test_images)} images from {len(test_ids)} vehicles")
    print(f"Query: {len(query_images)} images")

def download_pretrained_models():
    """
    Download and setup pretrained models for vehicle re-ID
    """
    import urllib.request
    import gdown
    
    model_dir = Path('pretrained_models')
    model_dir.mkdir(exist_ok=True)
    
    print("Downloading pretrained models...")
    
    # OSNet pretrained on ImageNet + VehicleID
    osnet_url = "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_imagenet.pth"
    osnet_path = model_dir / "osnet_x1_0_imagenet.pth"
    
    if not osnet_path.exists():
        print("Downloading OSNet model...")
        urllib.request.urlretrieve(osnet_url, osnet_path)
        print(f"OSNet model saved to {osnet_path}")
    
    # ResNet50-IBN-a
    resnet_ibn_url = "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth"
    resnet_ibn_path = model_dir / "resnet50_ibn_a.pth"
    
    if not resnet_ibn_path.exists():
        print("Downloading ResNet50-IBN-a model...")
        urllib.request.urlretrieve(resnet_ibn_url, resnet_ibn_path)
        print(f"ResNet50-IBN-a model saved to {resnet_ibn_path}")
    
    # Vehicle-specific models (you'll need to train these or find pretrained ones)
    print("\nFor vehicle-specific pretrained models, you can:")
    print("1. Use the training script above to train on your data")
    print("2. Download from vehicle re-ID repositories:")
    print("   - VeRi-776 dataset: https://vehiclereid.github.io/VeRi/")
    print("   - VehicleID dataset: https://www.pkuml.org/resources/pku-vehicleid.html")
    print("   - CityFlow dataset: https://www.aicitychallenge.org/")
    
    return {
        'osnet': str(osnet_path),
        'resnet_ibn': str(resnet_ibn_path)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Vehicle Re-ID Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--model_type', type=str, default='osnet',
                       choices=['osnet', 'resnet_ibn', 'efficientnet'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save models')
    parser.add_argument('--create_splits', action='store_true',
                       help='Create train/val/test splits from images')
    parser.add_argument('--download_pretrained', action='store_true',
                       help='Download pretrained models')
    
    args = parser.parse_args()
    
    if args.download_pretrained:
        download_pretrained_models()
    
    if args.create_splits:
        create_dataset_splits(args.data_dir)
    
    # Train model
    train_vehicle_reid_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )