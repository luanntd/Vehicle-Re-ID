import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from utils.dataset import VehicleReIDDataset
from utils.losses import TripletLoss, CenterLoss
from utils.compute_metrics import compute_mAP
from modules.feature_extraction import OSNet, ResNetIBN, create_vehicle_descriptor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    data_dir,
    image_dir,
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
    train_dataset = VehicleReIDDataset(data_dir, image_dir, split='train', transform=train_transform)
    val_dataset = VehicleReIDDataset(data_dir, image_dir, split='val', transform=val_transform)
    
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
        descriptor = create_vehicle_descriptor('efficientnet', num_classes=train_dataset.num_classes)
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
            mAP = evaluate(model, val_loader, model_type)
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

def evaluate(model, data_loader, model_type):
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
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                features = outputs[0]
            else:
                features = outputs
            
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu())
            all_labels.extend(labels.numpy())
            all_camera_ids.extend(camera_ids.numpy())
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)
    all_camera_ids = np.array(all_camera_ids)
    
    # Compute mAP
    mAP = compute_mAP(all_features, all_labels, all_camera_ids, use_cosine=True)
    
    return mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Vehicle Re-ID Model')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--image_dir', type=str, default='data/images',
                       help='Path to image directory')
    parser.add_argument('--model_type', type=str, default='osnet',
                       choices=['osnet', 'resnet_ibn', 'efficientnet'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    
    # Train model
    train(
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        model_type=args.model_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
