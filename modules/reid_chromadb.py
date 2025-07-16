import torch
import chromadb
from chromadb.config import Settings
import numpy as np
import cv2
import os
import base64
from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChromaDBVehicleReID:
    def __init__(self, 
                 db_path: str = "./chroma_vehicle_reid", 
                 collection_name: str = "vehicle_embeddings"):
        """
        Initialize the ChromaDB-based vehicle re-identification system.
        
        Parameters
        ----------
        db_path: str
            Path to ChromaDB database directory
        collection_name: str
            Name of the ChromaDB collection
        """
        # Confidence thresholds for different scenarios
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
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection for each vehicle type
        self.collections = {}
        for vehicle_type_id, vehicle_type_name in self.VEHICLE_TYPES.items():
            collection_name_full = f"{collection_name}_{vehicle_type_name}"
            try:
                self.collections[vehicle_type_id] = self.client.get_collection(
                    name=collection_name_full
                )
                print(f"Loaded existing collection: {collection_name_full}")
            except:
                # Collection doesn't exist, create it
                self.collections[vehicle_type_id] = self.client.create_collection(
                    name=collection_name_full,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                print(f"Created new collection: {collection_name_full}")
        
        # Track current max IDs for each vehicle type
        self.current_max_ids = self._load_max_ids()
    
    def _load_max_ids(self) -> Dict[int, int]:
        """Load current maximum IDs for each vehicle type from database."""
        max_ids = {}
        for vehicle_type_id in self.VEHICLE_TYPES.keys():
            collection = self.collections[vehicle_type_id]
            # Get all documents to find max ID
            try:
                results = collection.get()
                if results['metadatas']:
                    vehicle_ids = [int(meta['vehicle_id']) for meta in results['metadatas']]
                    max_ids[vehicle_type_id] = max(vehicle_ids) + 1 if vehicle_ids else 0
                else:
                    max_ids[vehicle_type_id] = 0
            except Exception as e:
                print(f"Error loading max IDs for vehicle type {vehicle_type_id}: {e}")
                max_ids[vehicle_type_id] = 0
        
        return max_ids
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode image to base64 string for storage."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _decode_image(self, encoded_image: str) -> np.ndarray:
        """Decode base64 string back to image."""
        image_data = base64.b64decode(encoded_image)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def calculate_similarity(self, 
                           target: torch.Tensor, 
                           vehicle_type: int,
                           top_k: int = 50) -> Tuple[List[float], List[Dict]]:
        """
        Calculate similarity between target and stored embeddings.
        
        Parameters
        ----------
        target: torch.Tensor
            Target embedding vector
        vehicle_type: int
            Vehicle type (0: motorcycle, 1: car, 2: truck, 3: bus)
        top_k: int
            Number of top similar embeddings to return
            
        Returns
        -------
        Tuple[List[float], List[Dict]]: Similarities and metadata
        """
        collection = self.collections[vehicle_type]
        
        # Convert tensor to list for ChromaDB
        target_embedding = target.cpu().numpy().tolist()
        
        try:
            # Query similar embeddings
            results = collection.query(
                query_embeddings=[target_embedding],
                n_results=min(top_k, collection.count()),
                include=['distances', 'metadatas', 'embeddings']
            )
            
            if not results['distances'][0]:
                return [], []
            
            # Convert distances to similarities (ChromaDB returns cosine distances)
            # Cosine similarity = 1 - cosine distance
            similarities = [1 - dist for dist in results['distances'][0]]
            metadatas = results['metadatas'][0]
            
            return similarities, metadatas
            
        except Exception as e:
            print(f"Error querying embeddings: {e}")
            return [], []
    
    def identify(self,
                target: torch.Tensor,
                vehicle_type: int,
                confidence: float = 1.0,
                do_update: bool = False,
                image: Optional[np.ndarray] = None,
                camera_id: Optional[str] = None,
                track_id: Optional[int] = None,
                timestamp: Optional[str] = None) -> int:
        """
        Identify a vehicle based on its features.
        
        Parameters
        ----------
        target: torch.Tensor
            Feature vector of the vehicle to identify
        vehicle_type: int
            Class index of the vehicle (0: motorcycle, 1: car, 2: truck, 3: bus)
        confidence: float
            Detection confidence from YOLO
        do_update: bool
            Whether to update the embeddings database
        image: np.ndarray, optional
            Vehicle image for storage
        camera_id: str, optional
            Camera identifier
        track_id: int, optional
            Track ID from YOLO
        timestamp: str, optional
            Timestamp of detection
            
        Returns
        -------
        int: Unique ID for the vehicle
        """
        # Default to new ID
        target_id = self.current_max_ids[vehicle_type]
        
        # Calculate similarity with existing vehicles of same type
        similarities, metadatas = self.calculate_similarity(target, vehicle_type)
        
        if similarities:
            # Get best match
            best_similarity = max(similarities)
            best_idx = similarities.index(best_similarity)
            
            # Adjust threshold based on detection confidence
            threshold = self.CONFIDENCE_THRESHOLD['normal']
            if confidence < 0.8:
                threshold = self.CONFIDENCE_THRESHOLD['low']
            elif confidence > 0.95:
                threshold = self.CONFIDENCE_THRESHOLD['high']
            
            # If good match found, use existing ID
            if best_similarity > threshold:
                target_id = int(metadatas[best_idx]['vehicle_id'])
                # print(f"Matched existing vehicle ID: {target_id} (similarity: {best_similarity:.3f})")
        
        # Update database if requested
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
    
    def _add_embedding(self,
                      target: torch.Tensor,
                      vehicle_type: int,
                      vehicle_id: int,
                      confidence: float,
                      image: Optional[np.ndarray] = None,
                      camera_id: Optional[str] = None,
                      track_id: Optional[int] = None,
                      timestamp: Optional[str] = None):
        """Add embedding and metadata to ChromaDB."""
        collection = self.collections[vehicle_type]
        
        # Normalize parameters to avoid None values
        camera_id = camera_id or 'unknown'
        track_id = track_id or -1
        
        # Check if a vehicle with the same vehicle_id, camera_id and track_id already exists
        # Only skip if both camera_id and track_id are valid values
        if camera_id != 'unknown' and track_id != -1:
            try:
                # Using proper operator format for ChromaDB where clause
                where_clause = {
                    "$and": [
                        {"vehicle_id": {"$eq": vehicle_id}},
                        {"camera_id": {"$eq": camera_id}}
                    ]
                }
                
                existing_records = collection.get(where=where_clause)
                
                # If this exact camera/track combination already exists for this vehicle_id,
                # don't add another duplicate record
                if existing_records and len(existing_records['ids']) > 0:
                    # print(f"Skipping duplicate for vehicle_id={vehicle_id}, camera={camera_id}, track={track_id}")
                    return
            except Exception as e:
                print(f"Error checking for existing records: {e}")
                # Continue with adding the embedding even if checking fails
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Convert embedding to list
        embedding = target.cpu().numpy().tolist()
        
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
        
        # Encode image if provided
        if image is not None:
            metadata['image_encoded'] = self._encode_image(image)
            metadata['image_shape'] = f"{image.shape[0]}x{image.shape[1]}x{image.shape[2]}"
        
        try:
            # Add to collection
            collection.add(
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
            )
            # print(f"Added embedding for vehicle ID {vehicle_id} to database")
        except Exception as e:
            print(f"Error adding embedding to database: {e}")
    
    def get_vehicle_history(self, vehicle_id: int, vehicle_type: int) -> List[Dict]:
        """
        Get all stored data for a specific vehicle ID.
        
        Parameters
        ----------
        vehicle_id: int
            Vehicle ID to search for
        vehicle_type: int
            Vehicle type
            
        Returns
        -------
        List[Dict]: All records for the vehicle
        """
        collection = self.collections[vehicle_type]
        
        try:
            # Use proper operator format for ChromaDB where clause
            where_clause = {"vehicle_id": {"$eq": vehicle_id}}
            
            results = collection.get(
                where=where_clause,
                include=['metadatas', 'embeddings']
            )
            
            history = []
            for i, metadata in enumerate(results['metadatas']):
                record = metadata.copy()
                # Decode image if present
                if 'image_encoded' in metadata:
                    record['image'] = self._decode_image(metadata['image_encoded'])
                    del record['image_encoded']  # Remove encoded version
                
                record['embedding'] = results['embeddings'][i]
                history.append(record)
            
            return history
            
        except Exception as e:
            print(f"Error retrieving vehicle history: {e}")
            return []
    
    def get_cross_camera_matches(self, vehicle_id: int, vehicle_type: int) -> Dict[str, List[Dict]]:
        """
        Get all camera appearances for a vehicle ID.
        
        Parameters
        ----------
        vehicle_id: int
            Vehicle ID to search for
        vehicle_type: int
            Vehicle type
            
        Returns
        -------
        Dict[str, List[Dict]]: Camera ID -> List of appearances
        """
        history = self.get_vehicle_history(vehicle_id, vehicle_type)
        
        # Group by camera
        camera_groups = {}
        for record in history:
            camera_id = record['camera_id']
            if camera_id not in camera_groups:
                camera_groups[camera_id] = []
            camera_groups[camera_id].append(record)
        
        return camera_groups
    
    def save_cross_camera_images(self, 
                                vehicle_id: int, 
                                vehicle_type: int, 
                                save_dir: str = "matching"):
        """
        Save images of vehicles that appear across multiple cameras.
        
        Parameters
        ----------
        vehicle_id: int
            Vehicle ID
        vehicle_type: int
            Vehicle type
        save_dir: str
            Directory to save images
        """
        camera_matches = self.get_cross_camera_matches(vehicle_id, vehicle_type)
        
        if len(camera_matches) > 1:
            vehicle_type_name = self.VEHICLE_TYPES[vehicle_type]
            vehicle_dir = f'{vehicle_type_name}_{vehicle_id}'
            match_dir = os.path.join(save_dir, vehicle_dir)
            
            if not os.path.exists(match_dir):
                os.makedirs(match_dir)
            
            # Save one image per unique track_id per camera
            for camera_id, records in camera_matches.items():
                saved_tracks = set()
                for record in records:
                    track_id = record['track_id']
                    key = (camera_id, track_id)
                    
                    if key not in saved_tracks and 'image' in record:
                        filename = f"{camera_id}_track{track_id}.jpg"
                        filepath = os.path.join(match_dir, filename)
                        cv2.imwrite(filepath, record['image'])
                        saved_tracks.add(key)
            
            print(f"Saved cross-camera images for vehicle {vehicle_id} in {match_dir}")
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats = {}
        total_embeddings = 0
        
        for vehicle_type_id, vehicle_type_name in self.VEHICLE_TYPES.items():
            collection = self.collections[vehicle_type_id]
            count = collection.count()
            stats[vehicle_type_name] = count
            total_embeddings += count
        
        stats['total'] = total_embeddings
        stats['max_ids'] = self.current_max_ids.copy()
        
        return stats
    
    def reset_database(self):
        """Reset the entire database - use with caution!"""
        for vehicle_type_id in self.VEHICLE_TYPES.keys():
            collection = self.collections[vehicle_type_id]
            # Delete all documents
            all_docs = collection.get()
            if all_docs['ids']:
                collection.delete(ids=all_docs['ids'])
        
        # Reset max IDs
        self.current_max_ids = {k: 0 for k in self.VEHICLE_TYPES.keys()}
        print("Database reset completed")
    
    def close(self):
        """Close database connection."""
        # ChromaDB client doesn't need explicit closing
        # Data is automatically persisted
        print("ChromaDB connection closed")