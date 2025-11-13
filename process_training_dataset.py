#!/usr/bin/env python3
"""
Process all RGB images from the training dataset directory.
Detects pole tips and calculates 3D coordinates when depth maps are available.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from typing import Optional, Tuple


class DepthTo3DConverter:
    """Convert 2D pixel coordinates + depth to 3D world coordinates."""
    
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        """
        Initialize with camera intrinsic parameters.
        
        Args:
            fx: Focal length in x (pixels)
            fy: Focal length in y (pixels)
            cx: Principal point x-coordinate (pixels)
            cy: Principal point y-coordinate (pixels)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def pixel_to_3d(self, u: float, v: float, depth_mm: float) -> Optional[Tuple[float, float, float]]:
        """
        Convert pixel coordinates and depth to 3D world coordinates.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            depth_mm: Depth value in millimeters
            
        Returns:
            (x, y, z) in meters, or None if invalid depth
        """
        # Validate depth
        if not np.isfinite(depth_mm) or depth_mm <= 0:
            return None
        
        # Convert depth from millimeters to meters
        Z = depth_mm / 1000.0
        
        # Pinhole camera model: convert 2D pixel to 3D world coordinates
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        
        return (X, Y, Z)
    
    def bbox_to_3d(self, bbox_xyxy: np.ndarray, depth_map: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Convert bounding box center to 3D coordinates using depth map.
        Uses median depth in a 5x5 window around the center for robustness.
        
        Args:
            bbox_xyxy: Bounding box [x1, y1, x2, y2]
            depth_map: Depth map (float32, millimeters)
            
        Returns:
            (x, y, z) in meters, or None if invalid depth
        """
        # Calculate center of bounding box
        x1, y1, x2, y2 = bbox_xyxy
        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2
        
        # Extract 5x5 window around center
        u_int = int(round(center_u))
        v_int = int(round(center_v))
        
        window_size = 5
        half_window = window_size // 2
        
        u_start = max(0, u_int - half_window)
        u_end = min(depth_map.shape[1], u_int + half_window + 1)
        v_start = max(0, v_int - half_window)
        v_end = min(depth_map.shape[0], v_int + half_window + 1)
        
        depth_window = depth_map[v_start:v_end, u_start:u_end]
        
        # Get valid (finite) depths
        valid_depths = depth_window[np.isfinite(depth_window)]
        
        if len(valid_depths) == 0:
            return None
        
        # Use median depth for robustness and cast to native float to satisfy type checkers
        median_depth = float(np.median(valid_depths))
        
        # Convert to 3D
        return self.pixel_to_3d(center_u, center_v, median_depth)


def find_depth_map(rgb_path: Path, depth_root: Path) -> Optional[Path]:
    """
    Find the corresponding depth map for an RGB image.
    Handles both original and augmented images.
    
    Args:
        rgb_path: Path to RGB image
        depth_root: Root directory containing depth maps
        
    Returns:
        Path to depth map, or None if not found
    """
    # Get the base name without augmentation suffix
    # e.g., "20250826_111213_front_frame000001_rgb_aug000.png" 
    #    -> "20250826_111213_front_frame000001"
    
    filename = rgb_path.stem  # Remove .png
    
    # Remove "_rgb" suffix
    if filename.endswith('_rgb'):
        base_name = filename[:-4]
    else:
        # Remove augmentation suffix if present (e.g., _rgb_aug000)
        parts = filename.split('_')
        if 'aug' in parts[-1]:
            # Remove last part (aug000) and _rgb
            base_name = '_'.join(parts[:-2])
        elif parts[-1] == 'rgb':
            base_name = '_'.join(parts[:-1])
        else:
            base_name = filename
    
    # Construct depth map filename
    depth_filename = f"{base_name}_depth.tiff"
    
    # Search in both OG-framed-only and OG-no-frame directories
    for subdir in ['OG-framed-only', 'OG-no-frame']:
        depth_path = depth_root / subdir / depth_filename
        if depth_path.exists():
            return depth_path
    
    return None


def process_training_dataset(
    train_dir: str = 'data/4-yolo_dataset/images/train',
    depth_root: str = 'data/1-OG',
    model_path: str = 'runs/train/tip_detector3/weights/best.pt',
    output_json: str = 'training_dataset_3d_coords.json',
    conf_threshold: float = 0.01
):
    """
    Process all images in the training dataset directory.
    Detects pole tips and calculates 3D coordinates when depth maps are available.
    
    Args:
        train_dir: Directory containing training RGB images
        depth_root: Root directory containing depth maps
        model_path: Path to YOLO model weights
        output_json: Output JSON file path
        conf_threshold: Detection confidence threshold
    """
    
    print("="*70)
    print("Training Dataset 3D Coordinate Processing")
    print("="*70)
    
    # Initialize
    train_path = Path(train_dir)
    depth_root_path = Path(depth_root)
    
    # Load YOLO model
    print(f"\nLoading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # ZED X camera intrinsics (HD1080 mode)
    converter = DepthTo3DConverter(
        fx=1067.0,
        fy=1067.0,
        cx=960.0,
        cy=540.0
    )
    
    # Find all RGB images
    rgb_images = sorted(train_path.glob('*.png'))
    print(f"\nFound {len(rgb_images)} RGB images in {train_dir}")
    
    # Process images
    results_data = {
        'camera_intrinsics': {
            'fx': 1067.0,
            'fy': 1067.0,
            'cx': 960.0,
            'cy': 540.0,
            'resolution': [1920, 1080]
        },
        'summary': {
            'total_images': len(rgb_images),
            'images_with_detections': 0,
            'total_tips_detected': 0,
            'images_with_depth': 0,
            'tips_with_3d_coords': 0
        },
        'images': []
    }
    
    print("\nProcessing images...")
    print("-"*70)
    
    for idx, rgb_path in enumerate(rgb_images, 1):
        # Progress
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(rgb_images)} images processed...")
        
        # Find corresponding depth map
        depth_path = find_depth_map(rgb_path, depth_root_path)
        
        # Run detection
        results = model(str(rgb_path), conf=conf_threshold, verbose=False)
        
        # Extract detections
        boxes = results[0].boxes
        num_detections = len(boxes)
        
        if num_detections == 0:
            # No detections, skip
            continue
        
        results_data['summary']['images_with_detections'] += 1
        results_data['summary']['total_tips_detected'] += num_detections
        
        # Prepare image entry
        image_entry = {
            'image': rgb_path.name,
            'image_path': str(rgb_path),
            'depth_available': depth_path is not None,
            'depth_path': str(depth_path) if depth_path else None,
            'num_tips': num_detections,
            'tips': []
        }
        
        # Load depth map if available
        depth_map = None
        if depth_path:
            results_data['summary']['images_with_depth'] += 1
            depth_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        
        # Process each detection
        for tip_idx, box in enumerate(boxes):
            bbox_xyxy = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            
            # Calculate pixel coordinates (center of bbox)
            center_u = float((bbox_xyxy[0] + bbox_xyxy[2]) / 2)
            center_v = float((bbox_xyxy[1] + bbox_xyxy[3]) / 2)
            
            tip_entry = {
                'tip_id': tip_idx + 1,
                'confidence': confidence,
                'bbox': {
                    'x1': float(bbox_xyxy[0]),
                    'y1': float(bbox_xyxy[1]),
                    'x2': float(bbox_xyxy[2]),
                    'y2': float(bbox_xyxy[3])
                },
                'pixel_coords': {
                    'u': center_u,
                    'v': center_v
                },
                'world_coords_meters': None
            }
            
            # Calculate 3D coordinates if depth available
            if depth_map is not None:
                coords_3d = converter.bbox_to_3d(bbox_xyxy, depth_map)
                if coords_3d:
                    x, y, z = coords_3d
                    tip_entry['world_coords_meters'] = {
                        'x': x,
                        'y': y,
                        'z': z
                    }
                    results_data['summary']['tips_with_3d_coords'] += 1
            
            image_entry['tips'].append(tip_entry)
        
        results_data['images'].append(image_entry)
    
    # Summary
    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    
    summary = results_data['summary']
    print(f"\nTotal images scanned: {summary['total_images']}")
    print(f"Images with detections: {summary['images_with_detections']}")
    print(f"Total tips detected: {summary['total_tips_detected']}")
    print(f"Images with depth maps: {summary['images_with_depth']}")
    print(f"Tips with 3D coordinates: {summary['tips_with_3d_coords']}")
    
    if summary['total_tips_detected'] > 0:
        depth_coverage = (summary['images_with_depth'] / summary['images_with_detections']) * 100
        coord_success = (summary['tips_with_3d_coords'] / summary['total_tips_detected']) * 100
        print(f"\nDepth map coverage: {depth_coverage:.1f}%")
        print(f"3D coordinate success rate: {coord_success:.1f}%")
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_json}")
    
    # Show some examples
    if results_data['images']:
        print("\n" + "="*70)
        print("Example Detections:")
        print("="*70)
        
        for img in results_data['images'][:3]:  # Show first 3
            print(f"\nImage: {img['image']}")
            print(f"  Tips detected: {img['num_tips']}")
            print(f"  Depth available: {img['depth_available']}")
            
            for tip in img['tips']:
                pixel = tip['pixel_coords']
                print(f"  - Tip {tip['tip_id']}:")
                print(f"      Confidence: {tip['confidence']:.3f}")
                print(f"      Pixel: (u={pixel['u']:.1f}, v={pixel['v']:.1f})")
                
                if tip['world_coords_meters']:
                    world = tip['world_coords_meters']
                    print(f"      World: (x={world['x']:.3f}m, y={world['y']:.3f}m, z={world['z']:.3f}m)")
                else:
                    print(f"      World: No depth data")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    process_training_dataset()
