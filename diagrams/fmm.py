import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc
import matplotlib.gridspec as gridspec

def create_multipole_expansion():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Source particles
    particles = [(0.3, 0.5), (0.35, 0.45), (0.25, 0.55), (0.4, 0.5)]
    center = (0.3, 0.5)
    
    # Draw expansion circle
    ax.add_patch(Circle(center, 0.15, fill=False, color='blue', linestyle='--'))
    
    # Draw particles
    for p in particles:
        ax.add_patch(Circle(p, 0.02, color='red'))
    
    # Draw lines from center to particles
    for p in particles:
        ax.plot([center[0], p[0]], [center[1], p[1]], 'gray', alpha=0.5)
    
    # Draw multipole series representation
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.15
    x = center[0] + r*np.cos(theta)
    y = center[1] + r*np.sin(theta)
    ax.plot(x, y, 'green', alpha=0.5)
    
    # Labels
    ax.text(0.2, 0.7, 'Multipole\nExpansion', fontsize=12)
    ax.text(0.5, 0.5, 'z - z₀', fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multipole Expansion')
    
    plt.savefig('multipole_expansion.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_translation_operators():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw boxes representing different levels
    boxes = [(0.2, 0.4), (0.5, 0.4), (0.8, 0.4)]
    box_size = 0.15
    
    for i, center in enumerate(boxes):
        ax.add_patch(Rectangle((center[0]-box_size/2, center[1]-box_size/2), 
                             box_size, box_size, fill=False, color='blue'))
        
    # Draw arrows for translations
    ax.add_patch(FancyArrowPatch(boxes[0], boxes[1], 
                                arrowstyle='->',
                                connectionstyle='arc3,rad=0.2',
                                color='red',
                                label='M2M'))
                                
    ax.add_patch(FancyArrowPatch(boxes[1], boxes[2],
                                arrowstyle='->',
                                connectionstyle='arc3,rad=-0.2',
                                color='green',
                                label='M2L'))
    
    # Labels
    ax.text(0.3, 0.6, 'M2M Translation', color='red', fontsize=12)
    ax.text(0.6, 0.3, 'M2L Translation', color='green', fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('FMM Translation Operators')
    
    plt.savefig('fmm_translations.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interaction_list():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid of boxes
    grid_size = 3
    box_size = 0.2
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = 0.2 + i * box_size
            y = 0.2 + j * box_size
            ax.add_patch(Rectangle((x, y), box_size, box_size, 
                                 fill=False, color='gray'))
    
    # Highlight center box
    center_x = 0.2 + box_size
    center_y = 0.2 + box_size
    ax.add_patch(Rectangle((center_x, center_y), box_size, box_size,
                          fill=True, color='blue', alpha=0.3))
    
    # Highlight interaction list
    interaction_boxes = [
        (center_x-box_size, center_y+box_size),
        (center_x+box_size, center_y+box_size),
        (center_x+box_size, center_y-box_size),
        (center_x-box_size, center_y-box_size)
    ]
    
    for x, y in interaction_boxes:
        ax.add_patch(Rectangle((x, y), box_size, box_size,
                             fill=True, color='green', alpha=0.3))
    
    # Labels
    ax.text(center_x, center_y+0.1, 'Target\nBox', fontsize=10)
    ax.text(center_x-box_size, center_y+box_size+0.1, 'Interaction\nList', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('FMM Interaction List')
    
    plt.savefig('interaction_list.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all visualizations
create_multipole_expansion()
create_translation_operators()
create_interaction_list()