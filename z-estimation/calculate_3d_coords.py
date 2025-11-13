#!/usr/bin/env python3
"""
Calculate 3D coordinates from RGB images + Depth TIFF files.

Works with pre-captured images (not live camera):
- RGB: .png files (e.g., 20250826_111213_front_frame000001_rgb.png)
- Depth: .tiff files (e.g., 20250826_111213_front_frame000001_depth.tiff)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json


class DepthTo3DConverter:
    """Convert 2D pixel + depth → 3D coordinates."""
    
    def __init__(self, fx, fy, cx, cy):
        """
        Args:
            fx, fy: Focal lengths (pixels)
            cx, cy: Principal point (image center, pixels)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def pixel_to_3d(self, u, v, depth_mm):
        """
        Convert pixel (u, v) + depth → 3D coordinates.
        
        Args:
            u, v: Pixel coordinates
            depth_mm: Depth in millimeters
            
        Returns:
            (x, y, z) in meters, or (None, None, None) if invalid
        """
        # Check valid depth
        if not np.isfinite(depth_mm) or depth_mm <= 0:
            return None, None, None
        
        # Convert mm to meters
        depth_m = depth_mm / 1000.0
        
        # Calculate 3D coordinates (meters)
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        
        return x, y, z
    
    def bbox_to_3d(self, bbox_xyxy, depth_map):
        """
        Get 3D coordinates from bounding box center.
        
        Args:
            bbox_xyxy: [x1, y1, x2, y2]
            depth_map: Depth image (float32, millimeters)
            
        Returns:
            (x, y, z) in meters
        """
        x1, y1, x2, y2 = bbox_xyxy
        
        # Bounding box center
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        
        # Check bounds
        h, w = depth_map.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None, None, None
        
        # Sample 5×5 region around center (more robust)
        u_min = max(0, u - 2)
        u_max = min(w, u + 3)
        v_min = max(0, v - 2)
        v_max = min(h, v + 3)
        
        depth_region = depth_map[v_min:v_max, u_min:u_max]
        
        # Use median of valid depths
        valid_depths = depth_region[np.isfinite(depth_region) & (depth_region > 0)]
        
        if len(valid_depths) == 0:
            return None, None, None
        
        depth_mm = np.median(valid_depths)
        
        return self.pixel_to_3d(u, v, depth_mm)


def process_images(rgb_dir='data/1-OG/OG-framed-only',
                   output_json='tip_3d_coordinates.json',
                   conf_threshold=0.01):
    """
    Process RGB + depth TIFF pairs and calculate 3D tip positions.
    
    Args:
        rgb_dir: Directory with RGB images and depth TIFFs
        output_json: Output file
        conf_threshold: Detection confidence threshold
    """
    
    print("="*70)
    print("Tip 3D Coordinate Calculation")
    print("="*70)
    
    # ZED X camera intrinsics (1920×1200 HD1200 mode)
    # Depth TIFF is 1920×1080, so we need to adjust cy
    CAMERA_FX = 1067.0
    CAMERA_FY = 1067.0
    CAMERA_CX = 960.0   # width / 2
    CAMERA_CY = 540.0   # 1080 / 2 (not 600 for 1200!)
    
    print(f"\nCamera Intrinsics (for 1920×1080):")
    print(f"  fx = {CAMERA_FX}")
    print(f"  fy = {CAMERA_FY}")
    print(f"  cx = {CAMERA_CX}")
    print(f"  cy = {CAMERA_CY}")
    
    # Initialize
    model = YOLO('runs/train/tip_detector3/weights/best.pt')
    converter = DepthTo3DConverter(CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY)
    
    # Find RGB images
    rgb_path = Path(rgb_dir)
    rgb_files = sorted(rgb_path.glob('*_rgb.png'))
    
    if not rgb_files:
        print(f"\n❌ No RGB images found in {rgb_dir}")
        return
    
    print(f"\nFound {len(rgb_files)} RGB images")
    print(f"Confidence threshold: {conf_threshold}\n")
    
    # Process each image
    results_3d = []
    stats = {'total_images': 0, 'tips_detected': 0, 'tips_with_3d': 0}
    
    for rgb_file in rgb_files:
        # Find corresponding depth TIFF
        depth_file = rgb_file.parent / rgb_file.name.replace('_rgb.png', '_depth.tiff')
        
        if not depth_file.exists():
            print(f"⚠️  No depth file for {rgb_file.name}")
            continue
        
        # Load depth
        depth_map = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        
        if depth_map is None:
            print(f"⚠️  Failed to load {depth_file.name}")
            continue
        
        # Run detection
        result = model.predict(str(rgb_file), conf=conf_threshold, verbose=False)[0]
        
        stats['total_images'] += 1
        
        if result.boxes is None or len(result.boxes) == 0:
            print(f"  {rgb_file.name}: No tips detected")
            continue
        
        # Process detections
        image_data = {
            'image': rgb_file.name,
            'depth_file': depth_file.name,
            'tips': []
        }
        
        for i, box in enumerate(result.boxes):
            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # Get 3D coordinates
            x, y, z = converter.bbox_to_3d(bbox, depth_map)
            
            stats['tips_detected'] += 1
            
            tip_data = {
                'tip_id': i + 1,
                'confidence': round(conf, 3),
                'pixel_coords': {
                    'u': round((bbox[0] + bbox[2]) / 2, 1),
                    'v': round((bbox[1] + bbox[3]) / 2, 1)
                }
            }
            
            if x is not None:
                tip_data['world_coords_meters'] = {
                    'x': round(x, 3),
                    'y': round(y, 3),
                    'z': round(z, 3)
                }
                stats['tips_with_3d'] += 1
            else:
                tip_data['world_coords_meters'] = None
                tip_data['error'] = 'Invalid depth'
            
            image_data['tips'].append(tip_data)
        
        results_3d.append(image_data)
        
        # Print progress
        n_tips = len(image_data['tips'])
        n_valid = sum(1 for t in image_data['tips'] if t['world_coords_meters'])
        print(f"✓ {rgb_file.name}: {n_tips} tip(s), {n_valid} with 3D coords")
    
    # Save results
    output = {
        'camera_intrinsics': {
            'fx': CAMERA_FX,
            'fy': CAMERA_FY,
            'cx': CAMERA_CX,
            'cy': CAMERA_CY,
            'resolution': '1920x1080'
        },
        'summary': stats,
        'images': results_3d
    }
    
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"Images processed: {stats['total_images']}")
    print(f"Tips detected: {stats['tips_detected']}")
    print(f"Tips with 3D coords: {stats['tips_with_3d']}")
    
    if stats['tips_detected'] > 0:
        success_rate = stats['tips_with_3d'] / stats['tips_detected'] * 100
        print(f"3D success rate: {success_rate:.1f}%")
    
    print(f"\n✅ Results saved to: {output_json}")
    
    # Show example
    if results_3d and results_3d[0]['tips']:
        print("\n" + "="*70)
        print("Example Output:")
        print("="*70)
        example = results_3d[0]['tips'][0]
        print(f"Image: {results_3d[0]['image']}")
        print(f"Pixel: (u={example['pixel_coords']['u']}, v={example['pixel_coords']['v']})")
        if example['world_coords_meters']:
            coords = example['world_coords_meters']
            print(f"World: (x={coords['x']}m, y={coords['y']}m, z={coords['z']}m)")
            print(f"       → Tip is {coords['z']}m from camera")


if __name__ == '__main__':
    process_images(
        rgb_dir='data/1-OG/OG-framed-only',
        output_json='tip_3d_coordinates.json',
        conf_threshold=0.01
    )
