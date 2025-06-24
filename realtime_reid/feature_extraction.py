import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VehicleDescriptor:
    def __init__(self,
                 model_path=None,
                 input_size=(224, 224),
                 n_classes=1000):
        
        self.input_size = input_size
        
        # Initialize the model (using ResNet50 pretrained on vehicle dataset)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify the last layer for vehicle re-id
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, n_classes)
        
        # Load custom weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize transform pipeline
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def extract_feature(self, input_img: np.ndarray) -> torch.Tensor:
        """
        Extract features from a vehicle image.
        
        Parameters
        ----------
        input_img: np.ndarray
            The vehicle image to extract features from
            
        Returns
        -------
        torch.Tensor: Feature vector representing the vehicle
        """
        # Preprocess image
        img = self.transforms(input_img).unsqueeze(0)
        img = img.to(device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img)
            
        # Reshape and normalize features
        features = features.squeeze()
        features = nn.functional.normalize(features, p=2, dim=0)
        
        return features
