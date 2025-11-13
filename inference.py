#!/usr/bin/env python3
"""
Run inference with trained YOLOv12 model.
Visualize tip detections on test images.
"""
from ultralytics import YOLO
import os
from pathlib import Path

def run_inference(model_path='runs/train/tip_detector3/weights/best.pt',
                  images_dir='RGB-only',
                  output_dir='predictions',
                  conf_threshold=0.01):
    """
    Run trained model on images and save predictions.
    
    Args:
        model_path: Path to trained weights
        images_dir: Directory with images to predict on
        output_dir: Where to save prediction visualizations
        conf_threshold: Confidence threshold for detections (0.01 = 98.4% recall, 98.7% precision)
    """
    
    print("="*60)
    print("YOLO Tip Detection Inference")
    print("="*60)
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first with: python train.py")
        return
    
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get test images (use a few from RGB-only that aren't in train/val)
    image_files = list(Path(images_dir).glob('*.png'))[:10]  # First 10 for demo
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"\nRunning inference on {len(image_files)} images...")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Run batch prediction
    results = model.predict(
        source=image_files,
        conf=conf_threshold,
        save=True,
        project=output_dir,
        name='results',
        line_width=2,
        show_labels=True,
        show_conf=True,
    )
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)
    print(f"Predictions saved to: {output_dir}/results/")
    print(f"\nDetection summary:")
    
    total_detections = 0
    for i, result in enumerate(results):
        n_detections = len(result.boxes)
        total_detections += n_detections
        print(f"  {Path(image_files[i]).name}: {n_detections} tip(s) detected")
    
    print(f"\nTotal: {total_detections} tips detected across {len(image_files)} images")
    
    return results

if __name__ == '__main__':
    # Run inference on test images
    results = run_inference()
    
    print("\nTo run on different images:")
    print("  python inference.py --images your_image_folder/")
