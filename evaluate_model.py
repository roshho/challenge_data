#!/usr/bin/env python3
"""
Comprehensive evaluation of trained YOLO model on validation dataset.
"""
from ultralytics import YOLO
import argparse
from pathlib import Path

def evaluate_model(model_path='runs/train/tip_detector3/weights/best.pt', 
                   data_yaml='data.yaml',
                   save_plots=True):
    """
    Run full evaluation on validation dataset.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset configuration
        save_plots: Whether to save visualization plots
    """
    
    print("="*70)
    print("YOLO Model Evaluation")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # Get model info
    print(f"\nModel Information:")
    print(f"  Architecture: {model.model_name if hasattr(model, 'model_name') else 'YOLOv8'}")
    model_size = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"  Model size: {model_size:.1f} MB")
    
    # Run validation
    print(f"\nRunning validation on dataset: {data_yaml}")
    print("This will evaluate:")
    print("  - Precision, Recall, mAP@0.5, mAP@0.5:0.95")
    print("  - Per-class performance")
    print("  - Confusion matrix")
    print("  - Precision-Recall curves")
    print("  - F1-Confidence curves")
    print("\nValidating...")
    
    metrics = model.val(
        data=data_yaml,
        split='val',
        plots=save_plots,
        save_json=False,
        verbose=True,
        conf=0.001,  # Low confidence threshold to see all detections
        iou=0.6,     # IoU threshold for NMS
    )
    
    # Print detailed results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Box metrics
    box_metrics = metrics.box
    print(f"\n📊 Detection Metrics:")
    print(f"  Precision (P):     {box_metrics.mp:.4f}  (What % of predictions are correct)")
    print(f"  Recall (R):        {box_metrics.mr:.4f}  (What % of ground truth found)")
    print(f"  mAP@0.5:           {box_metrics.map50:.4f}  (Mean Average Precision at 50% IoU)")
    print(f"  mAP@0.5:0.95:      {box_metrics.map:.4f}  (mAP across IoU 50-95%)")
    print(f"  F1-Score:          {2 * (box_metrics.mp * box_metrics.mr) / (box_metrics.mp + box_metrics.mr) if (box_metrics.mp + box_metrics.mr) > 0 else 0:.4f}  (Harmonic mean of P and R)")
    
    # Per-class metrics
    print(f"\n📈 Per-Class Performance:")
    if hasattr(box_metrics, 'ap_class_index'):
        class_names = metrics.names
        for i, class_idx in enumerate(box_metrics.ap_class_index):
            class_name = class_names[class_idx]
            print(f"  Class '{class_name}':")
            if hasattr(box_metrics, 'p') and len(box_metrics.p) > i:
                print(f"    Precision:  {box_metrics.p[i]:.4f}")
            if hasattr(box_metrics, 'r') and len(box_metrics.r) > i:
                print(f"    Recall:     {box_metrics.r[i]:.4f}")
            if hasattr(box_metrics, 'ap50') and len(box_metrics.ap50) > i:
                print(f"    AP@0.5:     {box_metrics.ap50[i]:.4f}")
            if hasattr(box_metrics, 'ap') and len(box_metrics.ap) > i:
                print(f"    AP@0.5:0.95: {box_metrics.ap[i]:.4f}")
    
    # Inference speed
    print(f"\n⚡ Inference Speed:")
    speed = metrics.speed
    print(f"  Preprocess:  {speed['preprocess']:.1f} ms")
    print(f"  Inference:   {speed['inference']:.1f} ms")
    print(f"  Postprocess: {speed['postprocess']:.1f} ms")
    total_time = sum(speed.values())
    fps = 1000 / total_time if total_time > 0 else 0
    print(f"  Total:       {total_time:.1f} ms ({fps:.1f} FPS)")
    
    # Dataset info
    print(f"\n📁 Dataset Information:")
    print(f"  Validation images: {metrics.seen}")
    print(f"  Total labels: {metrics.nt_per_class.sum() if hasattr(metrics, 'nt_per_class') else 'N/A'}")
    
    # Save location
    if save_plots:
        save_dir = Path(model_path).parent.parent / 'val'
        print(f"\n💾 Validation plots saved to: {save_dir}/")
        print(f"  - confusion_matrix.png")
        print(f"  - BoxPR_curve.png")
        print(f"  - BoxF1_curve.png")
        print(f"  - val_batch*_pred.jpg")
    
    # Performance interpretation
    print("\n" + "="*70)
    print("PERFORMANCE INTERPRETATION")
    print("="*70)
    
    map50 = box_metrics.map50
    precision = box_metrics.mp
    recall = box_metrics.mr
    
    print(f"\n🎯 Overall Grade: ", end="")
    if map50 >= 0.95:
        print("EXCELLENT ⭐⭐⭐⭐⭐")
    elif map50 >= 0.90:
        print("VERY GOOD ⭐⭐⭐⭐")
    elif map50 >= 0.85:
        print("GOOD ⭐⭐⭐")
    elif map50 >= 0.80:
        print("FAIR ⭐⭐")
    else:
        print("NEEDS IMPROVEMENT ⭐")
    
    print(f"\n💡 Analysis:")
    if precision > 0.9 and recall < 0.85:
        print("  - High precision, lower recall = Model is conservative")
        print("  - Missing some tips but confident when it detects")
        print("  → Recommendation: Lower confidence threshold or add more training data")
    elif recall > 0.9 and precision < 0.85:
        print("  - High recall, lower precision = Model is aggressive")
        print("  - Finding most tips but some false positives")
        print("  → Recommendation: Raise confidence threshold or refine training")
    elif precision > 0.9 and recall > 0.9:
        print("  - Excellent balance! Model is both accurate and complete")
        print("  ✓ Ready for production use")
    else:
        print("  - Both metrics need improvement")
        print("  → Recommendation: More training data or longer training")
    
    print("\n" + "="*70)
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate YOLO model on validation set')
    parser.add_argument('--model', '-m', 
                       default='runs/train/tip_detector3/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--data', '-d',
                       default='data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip saving visualization plots')
    
    args = parser.parse_args()
    
    metrics = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        save_plots=not args.no_plots
    )
