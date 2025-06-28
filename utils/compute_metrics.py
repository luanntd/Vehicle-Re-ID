import torch
import numpy as np

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