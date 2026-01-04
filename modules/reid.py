import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import base64
from typing import Dict, List, Optional, Tuple
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VehicleReID:
    def __init__(self, from_file: str = None):
        self.CONFIDENCE_THRESHOLD = {
            'high': 0.95,    # For very confident matches
            'normal': 0.85,  # For regular matches
            'low': 0.75     # For partial/occluded vehicles
        }
        
        # Vehicle type mapping
        self.VEHICLE_TYPES = {
            0: 'motorcycle',
            1: 'car', 
            2: 'truck',
            3: 'bus'
        }
        
        # Keys are vehicle class indices (0: motorcycle, 1: car, 2: truck, 3: bus)
        self.embeddings = {
            0: torch.Tensor().to(device),  # motorcycles
            1: torch.Tensor().to(device),  # cars
            2: torch.Tensor().to(device),  # trucks
            3: torch.Tensor().to(device)   # buses
        }
        
        # Initialize IDs for each vehicle type
        self.ids = {k: [] for k in self.embeddings.keys()}
        self.current_max_ids = {k: 0 for k in self.embeddings.keys()}
        
        # Store metadata for each embedding (camera_id, track_id, timestamp, image)
        self.metadatas = {k: [] for k in self.embeddings.keys()}
        
        # Load saved embeddings if provided
        if from_file:
            self.load_embeddings(from_file)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _decode_image(self, encoded_image: str) -> np.ndarray:
        """Decode base64 string to image"""
        image_data = base64.b64decode(encoded_image)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def calculate_similarity(self, target: torch.Tensor, vehicle_type: int, top_k: int = 50) -> Tuple[torch.Tensor, List[int]]:
        """Calculate cosine similarity between target and stored embeddings."""
        if self.embeddings[vehicle_type].shape[0] == 0:
            return torch.Tensor().to(device), []
        
        similarities = F.cosine_similarity(
            target.unsqueeze(0),
            self.embeddings[vehicle_type],
            dim=1
        )
        
        # Get top_k results
        k = min(top_k, similarities.shape[0])
        top_similarities, top_indices = torch.topk(similarities, k)
        
        return top_similarities, top_indices.cpu().tolist()

    def identify(
        self,
        target: torch.Tensor,
        vehicle_type: int,
        confidence: float = 1.0,
        do_update: bool = False,
        image: Optional[np.ndarray] = None,
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None,
        timestamp: Optional[str] = None
    ) -> int:
        # Default to new ID
        target_id = self.current_max_ids[vehicle_type]
        
        # Calculate similarity with existing vehicles of same type
        if self.embeddings[vehicle_type].shape[0] > 0:
            similarities, indices = self.calculate_similarity(target, vehicle_type)
            
            if len(similarities) > 0:
                # Get best match
                best_score = similarities[0].item()
                best_idx = indices[0]
                
                # Adjust threshold based on detection confidence
                threshold = self.CONFIDENCE_THRESHOLD['normal']
                if confidence < 0.8:
                    threshold = self.CONFIDENCE_THRESHOLD['low']
                elif confidence > 0.95:
                    threshold = self.CONFIDENCE_THRESHOLD['high']
                    
                # If good match found, use existing ID
                if best_score > threshold:
                    target_id = self.ids[vehicle_type][best_idx]
        
        # Update embeddings and metadata
        if do_update:
            self._add_embedding(
                target=target,
                vehicle_type=vehicle_type,
                vehicle_id=target_id,
                confidence=confidence,
                image=image,
                camera_id=camera_id,
                track_id=track_id,
                timestamp=timestamp
            )
            
            # Update max ID if new vehicle
            if target_id == self.current_max_ids[vehicle_type]:
                self.current_max_ids[vehicle_type] += 1
                
        return target_id
    
    def _add_embedding(
        self,
        target: torch.Tensor,
        vehicle_type: int,
        vehicle_id: int,
        confidence: float,
        image: Optional[np.ndarray] = None,
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None,
        timestamp: Optional[str] = None
    ):
        """Add embedding and metadata to memory"""
        # Normalize parameters
        camera_id = camera_id or 'unknown'
        track_id = track_id or -1
        
        # Check if a vehicle with the same vehicle_id and camera_id already exists
        if camera_id != 'unknown' and track_id != -1:
            for meta in self.metadatas[vehicle_type]:
                if meta['vehicle_id'] == vehicle_id and meta['camera_id'] == camera_id:
                    return  # Skip duplicate
        
        # Add embedding
        if self.embeddings[vehicle_type].shape[0] > 0:
            self.embeddings[vehicle_type] = torch.cat(
                (self.embeddings[vehicle_type], target.unsqueeze(0)),
                dim=0
            )
        else:
            # First vehicle of this type
            self.embeddings[vehicle_type] = target.unsqueeze(0)
        
        # Add ID
        self.ids[vehicle_type].append(vehicle_id)
        
        # Prepare metadata
        metadata = {
            'vehicle_id': vehicle_id,
            'vehicle_type': vehicle_type,
            'vehicle_type_name': self.VEHICLE_TYPES[vehicle_type],
            'confidence': confidence,
            'camera_id': camera_id,
            'track_id': track_id,
            'timestamp': timestamp or datetime.now().isoformat(),
        }
        
        # Encode and store image if provided
        if image is not None:
            metadata['image_encoded'] = self._encode_image(image)
            metadata['image_shape'] = f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}"
        
        self.metadatas[vehicle_type].append(metadata)
    
    def get_vehicle_history(self, vehicle_id: int, vehicle_type: int) -> List[Dict]:
        """Get all records for a specific vehicle"""
        history = []
        
        for i, meta in enumerate(self.metadatas[vehicle_type]):
            if meta['vehicle_id'] == vehicle_id:
                record = meta.copy()
                
                # Decode image if present
                if 'image_encoded' in record:
                    record['image'] = self._decode_image(record['image_encoded'])
                    del record['image_encoded']
                
                # Add embedding
                record['embedding'] = self.embeddings[vehicle_type][i].cpu().numpy().tolist()
                history.append(record)
        
        return history
    
    def get_cross_camera_matches(self, vehicle_id: int, vehicle_type: int) -> Dict[str, List[Dict]]:
        """Get vehicle appearances grouped by camera"""
        history = self.get_vehicle_history(vehicle_id, vehicle_type)
        
        # Group by camera
        camera_groups = {}
        for record in history:
            camera_id = record['camera_id']
            if camera_id not in camera_groups:
                camera_groups[camera_id] = []
            camera_groups[camera_id].append(record)
        
        return camera_groups
    
    def save_cross_camera_images(
        self,
        vehicle_id: int,
        vehicle_type: int,
        camera_matches,
        save_dir: str = "matching"
    ):
        """Save images of vehicle from different cameras"""
        vehicle_type_name = self.VEHICLE_TYPES[vehicle_type]
        vehicle_dir = f'{vehicle_type_name}_{vehicle_id}'
        match_dir = os.path.join(save_dir, vehicle_dir)
        
        if not os.path.exists(match_dir):
            os.makedirs(match_dir)
        
        # Save one image per unique track_id per camera
        for camera_id, records in camera_matches.items():
            for record in records:
                if 'image' in record:
                    track_id = record['track_id']
                    filename = f"{camera_id}_track{track_id}.jpg"
                    filepath = os.path.join(match_dir, filename)
                    cv2.imwrite(filepath, record['image'])
    
    def get_statistics(self) -> Dict:
        """Get statistics about stored embeddings"""
        stats = {}
        total_embeddings = 0
        
        for vehicle_type_id, vehicle_type_name in self.VEHICLE_TYPES.items():
            count = self.embeddings[vehicle_type_id].shape[0]
            stats[vehicle_type_name] = count
            total_embeddings += count
        
        stats['total'] = total_embeddings
        stats['max_ids'] = self.current_max_ids.copy()
        
        return stats
    
    def reset_database(self):
        """Reset all data in memory"""
        for vehicle_type_id in self.VEHICLE_TYPES.keys():
            self.embeddings[vehicle_type_id] = torch.Tensor().to(device)
            self.ids[vehicle_type_id] = []
            self.metadatas[vehicle_type_id] = []
        
        self.current_max_ids = {k: 0 for k in self.VEHICLE_TYPES.keys()}
        print("Database reset completed (in-memory)")

    def save_embeddings(self, filepath: str):
        """Save embeddings and metadata to disk"""
        torch.save({
            'embeddings': self.embeddings,
            'ids': self.ids,
            'metadatas': self.metadatas,
            'current_max_ids': self.current_max_ids
        }, filepath)
        print(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Load embeddings and metadata from disk"""
        saved_data = torch.load(filepath, map_location=device)
        self.embeddings = saved_data['embeddings']
        self.ids = saved_data['ids']
        self.current_max_ids = saved_data['current_max_ids']
        
        # Load metadatas if available (backward compatibility)
        if 'metadatas' in saved_data:
            self.metadatas = saved_data['metadatas']
        else:
            self.metadatas = {k: [] for k in self.embeddings.keys()}
        
        print(f"Loaded embeddings from {filepath}")
        stats = self.get_statistics()
        print(f"Total: {stats['total']} embeddings, Max IDs: {stats['max_ids']}")

