import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_mAP(features, labels, camera_ids, use_cosine=True):
    """Compute mean Average Precision for re-identification"""
    n = features.size(0)
    
    if use_cosine:
        # Cosine similarity (higher = more similar)
        similarity_matrix = torch.mm(features, features.t())
        # Convert to distance (lower = more similar) 
        dist_matrix = 1 - similarity_matrix
    else:
        # Euclidean distance (lower = more similar)
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


def compute_cmc_map(features, labels, camera_ids, ranks=[1, 3, 5], use_cosine=True):
    n = features.size(0)
    
    if use_cosine:
        # Cosine similarity (higher = more similar)
        similarity_matrix = torch.mm(features, features.t())
        # Convert to distance (lower = more similar) 
        dist_matrix = 1 - similarity_matrix
    else:
        # Euclidean distance (lower = more similar)
        dist_matrix = torch.cdist(features, features, p=2)
    
    aps = []
    cmc_scores = np.zeros(max(ranks))
    valid_queries = 0
    for i in range(n):
        query_label = labels[i]
        query_cam = camera_ids[i]
        distances = dist_matrix[i]
        valid_mask = (camera_ids != query_cam) & (np.arange(n) != i)
        if not valid_mask.any():
            continue
        gallery_labels = labels[valid_mask]
        gallery_distances = distances[valid_mask]
        sorted_indices = torch.argsort(gallery_distances)
        sorted_labels = gallery_labels[sorted_indices]
        matches = (sorted_labels == query_label)
        if not matches.any():
            continue
        # CMC
        first_match = np.where(matches)[0][0]
        for r in ranks:
            if first_match < r:
                cmc_scores[r-1] += 1
        # mAP
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
        valid_queries += 1
    cmc_scores = cmc_scores / valid_queries if valid_queries > 0 else cmc_scores
    mAP = np.mean(aps) if aps else 0.0
    return mAP, {f"Rank-{r}": cmc_scores[r-1] for r in ranks}

def compute_classification_metrics(features, labels, camera_ids, threshold=0.5):
    """
    Compute traditional classification metrics for vehicle re-identification using cosine similarity
    """
    n = features.size(0)
    
    # Cosine similarity (higher = more similar)
    similarity_matrix = torch.mm(features, features.t())
    distances = 1 - similarity_matrix  # Convert to distance
    
    # Collect predictions and ground truth for all query-gallery pairs
    y_true = []
    y_pred = []
    top1_correct = 0
    total_queries = 0
    
    for i in range(n):
        query_label = labels[i]
        query_cam = camera_ids[i]
        
        # Find valid gallery samples (different camera, exclude query)
        valid_mask = (camera_ids != query_cam) & (np.arange(n) != i)
        
        if not valid_mask.any():
            continue
        
        total_queries += 1
        gallery_labels = labels[valid_mask]
        gallery_distances = distances[i][valid_mask]
        
        # Top-1 accuracy: check if closest match is correct
        closest_idx = torch.argmin(gallery_distances)
        if gallery_labels[closest_idx] == query_label:
            top1_correct += 1
        
        # For threshold-based metrics
        for j, (gallery_label, dist) in enumerate(zip(gallery_labels, gallery_distances)):
            # Ground truth: 1 if same vehicle, 0 if different
            is_same_vehicle = 1 if gallery_label == query_label else 0
            y_true.append(is_same_vehicle)
            
            # Prediction: 1 if distance < threshold, 0 otherwise
            is_predicted_match = 1 if dist < threshold else 0
            y_pred.append(is_predicted_match)
    
    # Compute metrics
    top1_accuracy = top1_correct / total_queries if total_queries > 0 else 0.0
    
    if len(y_true) > 0:
        threshold_accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        threshold_accuracy = precision = recall = f1 = 0.0
    
    return {
        'top1_accuracy': top1_accuracy,
        'threshold_accuracy': threshold_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': threshold
    }