#!/usr/bin/env python3
"""
Train YOLOv12 on tip detection dataset.
Uses SGD optimizer (stochastic gradient descent) as requested.
"""
from ultralytics import YOLO
import torch

def train_yolo():
    print("="*60)
    print("YOLO Tip Detection Training")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU detected, training will be slow!")
    
    # Load model - using YOLOv8s (small) for better accuracy
    print("\nLoading YOLOv8s (small) pretrained model...")
    model = YOLO('yolov8s.pt')  # Larger model: 11M params vs 3M for nano
    
    print("\nStarting training with:")
    print("  - Optimizer: SGD (Stochastic Gradient Descent)")
    print("  - Epochs: 100")
    print("  - Image size: 640x640")
    print("  - Batch size: 16")
    print("  - Learning rate: default (0.01 for SGD)")
    print("  - Momentum: 0.937 (default)")
    print("  - Weight decay: 0.0005 (default)")
    
    # Determine device (GPU if available, otherwise CPU)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"  - Device: {device}")
    
    # Train
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        optimizer='SGD',  # Stochastic Gradient Descent
        device=device,  # Auto-select GPU or CPU
        project='runs/train',
        name='tip_detector',
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,  # Generate training plots
        val=True,  # Run validation
        verbose=True,
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Best model saved to: runs/train/tip_detector/weights/best.pt")
    print(f"Results saved to: runs/train/tip_detector/")
    print(f"\nFinal metrics:")
    # Safely access results.results_dict because results may be None or not have the attribute
    metrics = getattr(results, "results_dict", None)
    if isinstance(metrics, dict):
        mAP50 = metrics.get('metrics/mAP50(B)', 'N/A')
        mAP50_95 = metrics.get('metrics/mAP50-95(B)', 'N/A')
    else:
        mAP50 = 'N/A'
        mAP50_95 = 'N/A'
    print(f"  mAP@0.5: {mAP50}")
    print(f"  mAP@0.5:0.95: {mAP50_95}")
    
    return model, results

if __name__ == '__main__':
    model, results = train_yolo()
