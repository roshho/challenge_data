"""
Evaluate prediction accuracy by comparing OUTPUT-JSON results with ground truth labels.
This script compares predictions from RUN-ME.py against Label Studio annotations.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_ground_truth(json_path):
    """
    Parse Label Studio JSON format to extract tip keypoint locations.
    
    Returns:
        dict: {filename: {'x': float, 'y': float}} in pixel coordinates
    """
    with open(json_path, 'r') as f:
        tasks = json.load(f)
    
    ground_truth = {}
    
    for task in tasks:
        # Extract filename from file_upload (format: "hash-filename.png")
        file_upload = task.get('file_upload', '')
        if '-' in file_upload:
            filename = file_upload.split('-', 1)[1]
        else:
            filename = file_upload
        
        # Get annotations (should be only one per task)
        annotations = task.get('annotations', [])
        if not annotations:
            continue
        
        # Get the first annotation
        annotation = annotations[0]
        results = annotation.get('result', [])
        
        if not results:
            continue
        
        # Get keypoint location (normalized coordinates 0-100)
        result = results[0]
        value = result.get('value', {})
        
        # Get original image dimensions
        original_width = result.get('original_width', 1920)
        original_height = result.get('original_height', 1080)
        
        # Convert normalized coordinates (0-100) to pixel coordinates
        x_normalized = value.get('x', 0)  # percentage
        y_normalized = value.get('y', 0)  # percentage
        
        # Convert to pixel coordinates
        x_pixel = (x_normalized / 100.0) * original_width
        y_pixel = (y_normalized / 100.0) * original_height
        
        ground_truth[filename] = {
            'x': x_pixel,
            'y': y_pixel
        }
    
    return ground_truth


def parse_predictions(json_path):
    """
    Parse RUN-ME.py output JSON to extract predicted tip locations.
    
    Returns:
        dict: {filename: {'x': float, 'y': float, 'confidence': float}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    predictions = {}
    
    for image_data in data.get('images', []):
        filename = image_data.get('image', '')
        tips = image_data.get('tips', [])
        
        if not tips:
            predictions[filename] = None  # No detection
            continue
        
        # Use the first tip (highest confidence if sorted)
        tip = tips[0]
        pixel_coords = tip.get('pixel_coords', {})
        
        predictions[filename] = {
            'x': pixel_coords.get('u', 0),
            'y': pixel_coords.get('v', 0),
            'confidence': tip.get('confidence', 0)
        }
    
    return predictions


def calculate_euclidean_distance(pred, gt):
    """Calculate Euclidean distance between predicted and ground truth points."""
    return np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)


def evaluate_predictions(pred_json, gt_json, distance_threshold=50):
    """
    Evaluate predictions against ground truth.
    
    Args:
        pred_json: Path to predictions JSON (from RUN-ME.py)
        gt_json: Path to ground truth JSON (Label Studio format)
        distance_threshold: Pixel distance threshold for "correct" detection
    
    Returns:
        dict: Evaluation metrics
    """
    print("Loading ground truth labels...")
    ground_truth = parse_ground_truth(gt_json)
    print(f"  Loaded {len(ground_truth)} ground truth annotations")
    
    print("\nLoading predictions...")
    predictions = parse_predictions(pred_json)
    print(f"  Loaded {len(predictions)} predictions")
    
    # Find common images
    gt_files = set(ground_truth.keys())
    pred_files = set(predictions.keys())
    common_files = gt_files & pred_files
    
    print(f"\n{'='*70}")
    print(f"Dataset Overlap:")
    print(f"  Ground truth images: {len(gt_files)}")
    print(f"  Prediction images: {len(pred_files)}")
    print(f"  Common images: {len(common_files)}")
    print(f"  Missing from predictions: {len(gt_files - pred_files)}")
    print(f"  Extra in predictions: {len(pred_files - gt_files)}")
    
    if not common_files:
        print("\n⚠️  WARNING: No common images found!")
        print("   Ground truth filenames sample:", list(gt_files)[:5])
        print("   Prediction filenames sample:", list(pred_files)[:5])
        return None
    
    # Evaluation metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    distances = []
    
    # Per-image results
    results_per_image = []
    
    print(f"\n{'='*70}")
    print(f"Evaluating {len(common_files)} common images...")
    print(f"Distance threshold: {distance_threshold} pixels\n")
    
    for filename in sorted(common_files):
        gt = ground_truth[filename]
        pred = predictions[filename]
        
        if pred is None:
            # False negative: ground truth exists but no detection
            false_negatives += 1
            results_per_image.append({
                'filename': filename,
                'status': 'FN (No Detection)',
                'distance': None,
                'gt_x': gt['x'],
                'gt_y': gt['y'],
                'pred_x': None,
                'pred_y': None
            })
        else:
            # Detection exists, calculate distance
            distance = calculate_euclidean_distance(pred, gt)
            distances.append(distance)
            
            if distance <= distance_threshold:
                true_positives += 1
                status = 'TP (Correct)'
            else:
                false_positives += 1
                status = 'FP (Wrong Location)'
            
            results_per_image.append({
                'filename': filename,
                'status': status,
                'distance': distance,
                'gt_x': gt['x'],
                'gt_y': gt['y'],
                'pred_x': pred['x'],
                'pred_y': pred['y'],
                'confidence': pred['confidence']
            })
    
    # Calculate metrics
    total = len(common_files)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Distance statistics
    if distances:
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
    else:
        mean_distance = median_distance = std_distance = min_distance = max_distance = None
    
    # Print summary
    print(f"{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Detection Metrics:")
    print(f"  True Positives (TP):  {true_positives:3d} / {total} ({true_positives/total*100:.1f}%)")
    print(f"  False Positives (FP): {false_positives:3d} / {total} ({false_positives/total*100:.1f}%)")
    print(f"  False Negatives (FN): {false_negatives:3d} / {total} ({false_negatives/total*100:.1f}%)")
    
    print(f"\nPerformance Metrics:")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    if distances:
        print(f"\nLocalization Error (pixels):")
        print(f"  Mean:   {mean_distance:.2f} px")
        print(f"  Median: {median_distance:.2f} px")
        print(f"  Std:    {std_distance:.2f} px")
        print(f"  Min:    {min_distance:.2f} px")
        print(f"  Max:    {max_distance:.2f} px")
    
    # Show worst cases
    print(f"\n{'='*70}")
    print(f"Top 10 Worst Localization Errors:")
    print(f"{'='*70}")
    
    worst_cases = sorted([r for r in results_per_image if r['distance'] is not None], 
                         key=lambda x: x['distance'], reverse=True)[:10]
    
    for i, case in enumerate(worst_cases, 1):
        print(f"{i:2d}. {case['filename']}")
        print(f"    Distance: {case['distance']:.2f} px")
        print(f"    Ground Truth: ({case['gt_x']:.1f}, {case['gt_y']:.1f})")
        print(f"    Predicted:    ({case['pred_x']:.1f}, {case['pred_y']:.1f})")
        print(f"    Confidence:   {case['confidence']:.4f}")
    
    # Show false negatives
    false_neg_cases = [r for r in results_per_image if r['status'] == 'FN (No Detection)']
    if false_neg_cases:
        print(f"\n{'='*70}")
        print(f"False Negatives (No Detection):")
        print(f"{'='*70}")
        for i, case in enumerate(false_neg_cases[:10], 1):
            print(f"{i:2d}. {case['filename']}")
            print(f"    Ground Truth: ({case['gt_x']:.1f}, {case['gt_y']:.1f})")
    
    # Return metrics
    return {
        'total_images': total,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_distance_px': mean_distance,
        'median_distance_px': median_distance,
        'std_distance_px': std_distance,
        'min_distance_px': min_distance,
        'max_distance_px': max_distance,
        'results_per_image': results_per_image
    }


if __name__ == '__main__':
    # Paths
    predictions_json = 'OUTPUT-JSON/INPUT-RGBD.json'
    ground_truth_json = 'data/2-RGB_only_depth_removed/labelled-data.json'
    
    # Check if files exist
    if not Path(predictions_json).exists():
        print(f"❌ Error: Predictions file not found: {predictions_json}")
        exit(1)
    
    if not Path(ground_truth_json).exists():
        print(f"❌ Error: Ground truth file not found: {ground_truth_json}")
        exit(1)
    
    # Run evaluation
    metrics = evaluate_predictions(
        predictions_json,
        ground_truth_json,
        distance_threshold=50  # 50 pixels = acceptable error
    )
    
    if metrics:
        # Save detailed results to JSON
        output_path = 'OUTPUT-JSON/evaluation_results.json'
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Detailed results saved to: {output_path}")
