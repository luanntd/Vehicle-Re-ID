import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import VehicleReIDDataset
from utils.compute_metrics import compute_cmc_map
from realtime_reid.feature_extraction import VehicleDescriptor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_reid(model, loader, args):
    """Test feature extraction + similarity ranking"""
    print("\n=== Testing Feature Extraction + Similarity Ranking ===")
    
    # Extract features
    all_features = []
    all_labels = []
    all_camera_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting features'):
            images = batch['image'].to(device)
            outputs = model(images)
            
            # Extract features only
            if isinstance(outputs, tuple):
                features = outputs[0]  # Use features, ignore logits
            else:
                features = outputs
            
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_features.append(features.cpu())
            all_labels.extend(batch['label'].numpy())
            all_camera_ids.extend(batch['camera_id'].numpy())
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)
    all_camera_ids = np.array(all_camera_ids)

    # Test with cosine similarity (primary method)
    mAP_cos, cmc_cos = compute_cmc_map(all_features, all_labels, all_camera_ids, ranks=[1, 3, 5], use_cosine=True)
    print(f"Cosine Similarity - mAP: {mAP_cos:.4f}")
    for rank, score in cmc_cos.items():
        print(f"Cosine Similarity - {rank}: {score:.4f}")
    
    # Test with Euclidean distance for comparison
    mAP_eucl, cmc_eucl = compute_cmc_map(all_features, all_labels, all_camera_ids, ranks=[1, 3, 5], use_cosine=False)
    print(f"Euclidean Distance - mAP: {mAP_eucl:.4f}")
    for rank, score in cmc_eucl.items():
        print(f"Euclidean Distance - {rank}: {score:.4f}")
    
    return mAP_cos, cmc_cos

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Vehicle Re-ID Model on images_test')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset root (should contain images_test)')
    parser.add_argument('--model_type', type=str, default='osnet', choices=['osnet', 'resnet_ibn', 'efficientnet'], help='Model type')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_osnet_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    args = parser.parse_args()

    # Define transforms (same as val)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare test data from images_test directory
    images_test_dir = os.path.join(args.data_dir, 'images_test')
    if not os.path.exists(images_test_dir):
        print(f"Error: {images_test_dir} does not exist!")
        return
    
    image_files = [f for f in os.listdir(images_test_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if not image_files:
        print(f"No image files found in {images_test_dir}")
        return
    
    print(f"Found {len(image_files)} test images in {images_test_dir}")
    
    # Create test split file
    split_file = os.path.join(args.data_dir, 'test.txt')
    with open(split_file, 'w') as f:
        for img in image_files:
            f.write(f"{img}\n")

    # Dataset and loader
    dataset = VehicleReIDDataset(args.data_dir, split='test', transform=val_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset loaded: {len(dataset)} samples, {dataset.num_classes} unique vehicles")

    # Create VehicleDescriptor (handles model creation and loading automatically)
    descriptor = VehicleDescriptor(
        model_path=args.model_path,
        model_type=args.model_type,
        num_classes=dataset.num_classes
    )
    
    model = descriptor.model
    print(f"Model info: {descriptor.get_model_info()}")
    
    model.eval()

    # Run feature extraction test
    test_reid(model, loader, args)

if __name__ == "__main__":
    main()
