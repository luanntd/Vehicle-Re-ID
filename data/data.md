# Dataset Configuration for Training and Testing
Expected data format:
```
dataset/
├── images/
│   ├── car_0001_cam1.jpg  # Format: vehicleType_vehicleID_cameraID.jpg
│   ├── car_0001_cam2.jpg
│   ├── car_0002_cam1.jpg
│   └── ...
├── train.txt              # List of training image file names
├── test.txt               # List of test image file names
└── query.txt              # List of query image file names
```