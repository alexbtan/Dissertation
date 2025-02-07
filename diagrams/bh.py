import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import matplotlib.gridspec as gridspec

def create_quadtree_division():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Main square
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, color='black'))
    
    # Quadrant divisions
    ax.axvline(x=0.5, color='black', linestyle='-', alpha=0.5)
    ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
    
    # Subdivided quadrant (top right)
    ax.add_patch(Rectangle((0.5, 0.5), 0.25, 0.25, fill=False, color='black', alpha=0.5))
    ax.add_patch(Rectangle((0.75, 0.5), 0.25, 0.25, fill=False, color='black', alpha=0.5))
    
    # Add some particles
    particles = [(0.2, 0.3), (0.8, 0.7), (0.6, 0.6), (0.9, 0.6), (0.3, 0.8)]
    for p in particles:
        ax.add_patch(Circle(p, 0.02, color='blue'))
    
    # Labels
    ax.text(0.25, 0.75, 'NW', fontsize=12)
    ax.text(0.75, 0.75, 'NE', fontsize=12)
    ax.text(0.25, 0.25, 'SW', fontsize=12)
    ax.text(0.75, 0.25, 'SE', fontsize=12)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('QuadTree Space Division')
    
    plt.savefig('quadtree_division.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_force_approximation():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Target particle
    target = (0.2, 0.5)
    ax.add_patch(Circle(target, 0.02, color='red'))
    ax.text(0.15, 0.55, 'Target', fontsize=12)
    
    # Distant cluster
    cluster_center = (0.7, 0.5)
    cluster_size = 0.15
    ax.add_patch(Rectangle((cluster_center[0]-cluster_size, cluster_center[1]-cluster_size), 
                          2*cluster_size, 2*cluster_size, fill=False, color='blue', 
                          linestyle='--'))
    
    # Particles in cluster
    particles = [(0.65, 0.45), (0.7, 0.52), (0.75, 0.48)]
    for p in particles:
        ax.add_patch(Circle(p, 0.01, color='blue'))
    
    # Center of mass
    ax.add_patch(Circle(cluster_center, 0.02, color='purple'))
    ax.text(0.72, 0.55, 'COM', fontsize=12)
    
    # Theta angle visualization
    theta_radius = 0.1
    ax.add_patch(plt.matplotlib.patches.Arc(target, theta_radius, theta_radius, 
                                          theta1=0, theta2=30, color='gray'))
    ax.text(0.25, 0.53, 'θ', fontsize=12)
    
    # Force arrow
    ax.add_patch(FancyArrowPatch(target, (0.5, 0.5), color='green',
                                arrowstyle='->',
                                mutation_scale=20))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Barnes-Hut Force Approximation')
    
    plt.savefig('force_approximation.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate both visualizations
create_quadtree_division()
create_force_approximation()