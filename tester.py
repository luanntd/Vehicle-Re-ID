import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import VehicleReIDDataset
from utils.compute_metrics import compute_cmc_map, compute_classification_metrics
from modules.feature_extraction import VehicleDescriptor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, loader, args):
    """Test feature extraction + similarity ranking with comprehensive metrics using cosine similarity
    """
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

    # Compute Re-ID metrics using cosine similarity
    mAP, cmc = compute_cmc_map(all_features, all_labels, all_camera_ids, ranks=[1, 3, 5], use_cosine=True)
    print(f"mAP: {mAP:.4f}")
    for rank, score in cmc.items():
        print(f"{rank}: {score:.4f}")

    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(all_labels)}")
    print(f"Unique vehicles: {len(np.unique(all_labels))}")
    print(f"Unique cameras: {len(np.unique(all_camera_ids))}")

    # 2. Classification metrics with different thresholds
    print("\n=== Classification Metrics ===")
    
    thresholds = args.thresholds if hasattr(args, 'thresholds') else [0.5, 0.8]
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Cosine similarity based metrics
        cos_metrics = compute_classification_metrics(
            all_features, all_labels, all_camera_ids, 
            threshold=threshold
        )
        
        print(f"\nThreshold={threshold}:")
        print(f"  Top-1 Accuracy: {cos_metrics['top1_accuracy']:.4f}")
        print(f"  Threshold Accuracy: {cos_metrics['threshold_accuracy']:.4f}")
        print(f"  Precision: {cos_metrics['precision']:.4f}")
        print(f"  Recall: {cos_metrics['recall']:.4f}")
        print(f"  F1-Score: {cos_metrics['f1_score']:.4f}")
        
        if cos_metrics['f1_score'] > best_f1:
            best_f1 = cos_metrics['f1_score']
            best_threshold = threshold

    print(f"\nBest F1-Score: {best_f1:.4f} at threshold={best_threshold}")

    # Summary
    print("\n=== Summary ===")
    print(f"Re-ID Performance:")
    print(f"  mAP: {mAP:.4f}")
    print(f"  Rank-1: {cmc['Rank-1']:.4f}")
    print(f"Classification Performance:")
    print(f"  Best F1-Score: {best_f1:.4f}")
    
    return {
        'mAP': mAP,
        'cmc': cmc,
        'best_f1': best_f1,
        'best_threshold': best_threshold
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Vehicle Re-ID Model on images_test')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset root (should contain images_test)')
    parser.add_argument('--image_dir', type=str, default='data/images_test', help='Path to image directory')
    parser.add_argument('--model_type', type=str, default='osnet', choices=['osnet', 'resnet_ibn', 'efficientnet'], help='Model type')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_osnet_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--save_results', type=str, default=None, help='Path to save detailed results (JSON format)')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.5, 0.8], help='Similarity thresholds for classification metrics')
    args = parser.parse_args()

    # Define transforms (same as val)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare test data from images_test directory
    images_test_dir = args.image_dir
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
    dataset = VehicleReIDDataset(args.data_dir, args.image_dir, split='test', transform=val_transform)
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
    results = test(model, loader, args)
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                        for k, v in value.items()}
                else:
                    json_results[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
            
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")

if __name__ == "__main__":
    main()
