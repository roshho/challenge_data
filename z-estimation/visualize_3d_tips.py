#!/usr/bin/env python3
"""
Visualize detected tip positions in 3D space.
"""

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def visualize_3d_tips(json_file='tip_3d_coordinates.json'):
    """
    Create 3D visualization of all detected tip positions.
    """
    
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("="*70)
    print("3D Tip Position Visualization")
    print("="*70)
    
    # Extract 3D coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    image_names = []
    
    for img_data in data['images']:
        for tip in img_data['tips']:
            if tip['world_coords_meters']:
                coords = tip['world_coords_meters']
                x_coords.append(coords['x'])
                y_coords.append(coords['y'])
                z_coords.append(coords['z'])
                image_names.append(img_data['image'])
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)
    
    print(f"\nTotal tips with 3D coordinates: {len(x_coords)}")
    print(f"\nCoordinate Ranges:")
    print(f"  X: {x_coords.min():.3f}m to {x_coords.max():.3f}m (span: {x_coords.max()-x_coords.min():.3f}m)")
    print(f"  Y: {y_coords.min():.3f}m to {y_coords.max():.3f}m (span: {y_coords.max()-y_coords.min():.3f}m)")
    print(f"  Z: {z_coords.min():.3f}m to {z_coords.max():.3f}m (span: {z_coords.max()-z_coords.min():.3f}m)")
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(14, 10))
    
    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(x_coords, y_coords, z_coords, 
                         c=z_coords, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='k')
    
    ax1.set_xlabel('X (meters) - Horizontal', fontsize=10)
    ax1.set_ylabel('Y (meters) - Vertical', fontsize=10)
    ax1.set_zlabel('Z (meters) - Depth', fontsize=10)
    ax1.set_title('3D Tip Positions (Color = Depth)', fontsize=12, fontweight='bold')
    
    # Add camera position at origin
    ax1.scatter([0], [0], [0], c='red', s=200, marker='^', 
               label='Camera', edgecolors='black', linewidths=2)
    ax1.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
    cbar.set_label('Depth (meters)', rotation=270, labelpad=15)
    
    # Set viewing angle
    ax1.view_init(elev=20, azim=45)
    
    # Top view (X-Z plane)
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter(x_coords, z_coords, 
                          c=z_coords, cmap='viridis', 
                          s=50, alpha=0.7, edgecolors='k')
    ax2.scatter([0], [0], c='red', s=200, marker='^', 
               edgecolors='black', linewidths=2, label='Camera')
    ax2.set_xlabel('X (meters) - Horizontal', fontsize=10)
    ax2.set_ylabel('Z (meters) - Depth', fontsize=10)
    ax2.set_title('Top View (X-Z Plane)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    # Side view (Y-Z plane)
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter(z_coords, y_coords, 
                          c=z_coords, cmap='viridis', 
                          s=50, alpha=0.7, edgecolors='k')
    ax3.scatter([0], [0], c='red', s=200, marker='^', 
               edgecolors='black', linewidths=2, label='Camera')
    ax3.set_xlabel('Z (meters) - Depth', fontsize=10)
    ax3.set_ylabel('Y (meters) - Vertical', fontsize=10)
    ax3.set_title('Side View (Y-Z Plane)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axis('equal')
    
    # Front view (X-Y plane)
    ax4 = fig.add_subplot(224)
    scatter4 = ax4.scatter(x_coords, y_coords, 
                          c=z_coords, cmap='viridis', 
                          s=50, alpha=0.7, edgecolors='k')
    ax4.scatter([0], [0], c='red', s=200, marker='^', 
               edgecolors='black', linewidths=2, label='Camera')
    ax4.set_xlabel('X (meters) - Horizontal', fontsize=10)
    ax4.set_ylabel('Y (meters) - Vertical', fontsize=10)
    ax4.set_title('Front View (X-Y Plane)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.savefig('tip_3d_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: tip_3d_visualization.png")
    
    # Create distance histogram
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distance histogram
    ax5.hist(z_coords, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(z_coords.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {z_coords.mean():.2f}m')
    ax5.set_xlabel('Distance from Camera (meters)', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Distribution of Tip Distances', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Horizontal position histogram
    ax6.hist(x_coords, bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax6.axvline(x_coords.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {x_coords.mean():.2f}m')
    ax6.set_xlabel('Horizontal Position (meters)', fontsize=11)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('Distribution of Horizontal Positions', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('tip_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Distribution plots saved to: tip_distribution.png")
    
    # Statistics
    print("\n" + "="*70)
    print("Statistics:")
    print("="*70)
    print(f"\nDepth (Z):")
    print(f"  Mean:   {z_coords.mean():.3f}m")
    print(f"  Median: {np.median(z_coords):.3f}m")
    print(f"  Std:    {z_coords.std():.3f}m")
    print(f"  Range:  {z_coords.min():.3f}m - {z_coords.max():.3f}m")
    
    print(f"\nHorizontal (X):")
    print(f"  Mean:   {x_coords.mean():.3f}m")
    print(f"  Median: {np.median(x_coords):.3f}m")
    print(f"  Std:    {x_coords.std():.3f}m")
    print(f"  Range:  {x_coords.min():.3f}m - {x_coords.max():.3f}m")
    
    print(f"\nVertical (Y):")
    print(f"  Mean:   {y_coords.mean():.3f}m")
    print(f"  Median: {np.median(y_coords):.3f}m")
    print(f"  Std:    {y_coords.std():.3f}m")
    print(f"  Range:  {y_coords.min():.3f}m - {y_coords.max():.3f}m")
    
    # Find extremes
    print("\n" + "="*70)
    print("Extreme Positions:")
    print("="*70)
    
    closest_idx = np.argmin(z_coords)
    farthest_idx = np.argmax(z_coords)
    
    print(f"\nClosest tip:")
    print(f"  Distance: {z_coords[closest_idx]:.3f}m")
    print(f"  Position: ({x_coords[closest_idx]:.3f}, {y_coords[closest_idx]:.3f}, {z_coords[closest_idx]:.3f})")
    print(f"  Image: {image_names[closest_idx]}")
    
    print(f"\nFarthest tip:")
    print(f"  Distance: {z_coords[farthest_idx]:.3f}m")
    print(f"  Position: ({x_coords[farthest_idx]:.3f}, {y_coords[farthest_idx]:.3f}, {z_coords[farthest_idx]:.3f})")
    print(f"  Image: {image_names[farthest_idx]}")
    
    leftmost_idx = np.argmin(x_coords)
    rightmost_idx = np.argmax(x_coords)
    
    print(f"\nLeftmost tip:")
    print(f"  X position: {x_coords[leftmost_idx]:.3f}m")
    print(f"  Image: {image_names[leftmost_idx]}")
    
    print(f"\nRightmost tip:")
    print(f"  X position: {x_coords[rightmost_idx]:.3f}m")
    print(f"  Image: {image_names[rightmost_idx]}")
    
    print("\n" + "="*70)
    print("✅ Visualization complete!")
    print("="*70)
    print("Generated files:")
    print("  - tip_3d_visualization.png (3D scatter + projections)")
    print("  - tip_distribution.png (histograms)")
    
    plt.show()


if __name__ == '__main__':
    visualize_3d_tips()
