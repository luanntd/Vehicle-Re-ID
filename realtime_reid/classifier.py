import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VehicleReID:
    def __init__(self, from_file: str = None):
        """
        Initialize the vehicle re-identification system.
        
        Parameters
        ----------
        from_file: str, optional
            Path to load saved embeddings from
        """
        # Confidence thresholds for different scenarios
        self.CONFIDENCE_THRESHOLD = {
            'high': 0.95,    # For very confident matches
            'normal': 0.85,  # For regular matches
            'low': 0.75     # For partial/occluded vehicles
        }
        
        # Initialize embeddings dictionary for different vehicle types
        # Keys are vehicle class indices (2: car, 3: motorcycle, 5: bus, 7: truck)
        self.embeddings = {
            2: torch.Tensor().to(device),  # cars
            3: torch.Tensor().to(device),  # motorcycles
            5: torch.Tensor().to(device),  # buses
            7: torch.Tensor().to(device)   # trucks
        }
        
        # Initialize IDs for each vehicle type
        self.ids = {k: [] for k in self.embeddings.keys()}
        self.current_max_ids = {k: 0 for k in self.embeddings.keys()}
        
        # Load saved embeddings if provided
        if from_file:
            saved_data = torch.load(from_file, map_location=device)
            self.embeddings = saved_data['embeddings']
            self.ids = saved_data['ids']
            self.current_max_ids = saved_data['current_max_ids']

    def calculate_similarity(self, target: torch.Tensor, vehicle_type: int) -> torch.Tensor:
        """Calculate cosine similarity between target and stored embeddings."""
        if self.embeddings[vehicle_type].shape[0] == 0:
            return torch.Tensor().to(device)
            
        return F.cosine_similarity(
            target.unsqueeze(0),
            self.embeddings[vehicle_type],
            dim=1
        )

    def identify(
        self,
        target: torch.Tensor,
        vehicle_type: int,
        confidence: float = 1.0,
        do_update: bool = False
    ) -> int:
        """
        Identify a vehicle based on its features.
        
        Parameters
        ----------
        target: torch.Tensor
            Feature vector of the vehicle to identify
        vehicle_type: int
            Class index of the vehicle (2: car, 3: motorcycle, 5: bus, 7: truck)
        confidence: float
            Detection confidence from YOLO
        do_update: bool
            Whether to update the embeddings database
            
        Returns
        -------
        int: Unique ID for the vehicle
        """
        # Default to new ID
        target_id = self.current_max_ids[vehicle_type]
        
        # Calculate similarity with existing vehicles of same type
        if self.embeddings[vehicle_type].shape[0] > 0:
            similarities = self.calculate_similarity(target, vehicle_type)
            
            # Get best match
            best_score, best_idx = torch.max(similarities, dim=0)
            best_score = best_score.item()
            
            # Adjust threshold based on detection confidence
            threshold = self.CONFIDENCE_THRESHOLD['normal']
            if confidence < 0.8:
                threshold = self.CONFIDENCE_THRESHOLD['low']
            elif confidence > 0.95:
                threshold = self.CONFIDENCE_THRESHOLD['high']
                
            # If good match found, use existing ID
            if best_score > threshold:
                target_id = self.ids[vehicle_type][best_idx]
            
            # Update embeddings
            if do_update:
                self.embeddings[vehicle_type] = torch.cat(
                    (self.embeddings[vehicle_type], target.unsqueeze(0)),
                    dim=0
                )
        else:
            # First vehicle of this type
            if do_update:
                self.embeddings[vehicle_type] = target.unsqueeze(0)
        
        # Update tracking info
        if do_update:
            self.ids[vehicle_type].append(target_id)
            if target_id == self.current_max_ids[vehicle_type]:
                self.current_max_ids[vehicle_type] += 1
                
        return target_id

    def save_embeddings(self, filepath: str):
        """Save the current embeddings and tracking info to a file."""
        torch.save({
            'embeddings': self.embeddings,
            'ids': self.ids,
            'current_max_ids': self.current_max_ids
        }, filepath)
            