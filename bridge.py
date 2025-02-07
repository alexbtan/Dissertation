import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

class Particle:
    def __init__(self, mass, position, velocity, fixed=False):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2)
        self.fixed = fixed
        self.history = [self.position.copy()]
        self.connections = []  # List of (particle, rest_length, spring_constant)

class QuadTreeNode:
    def __init__(self, center, size):
        self.center = np.array(center)
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(2)
        self.children = [None] * 4
        self.particle = None
        self.is_leaf = True

    def insert(self, particle):
        if self.particle is None and all(child is None for child in self.children):
            self.particle = particle
            self.mass = particle.mass
            self.com = particle.position.copy()
            return
        
        if self.particle is not None:
            self._split()
            self._insert_particle(self.particle)
            self.particle = None
            self._insert_particle(particle)
        else:
            self._insert_particle(particle)
        
        self.mass = sum(child.mass for child in self.children if child is not None)
        self.com = sum(child.com * child.mass for child in self.children if child is not None) / self.mass

    def _split(self):
        quarter_size = self.size / 4
        half_size = self.size / 2
        offsets = [[-quarter_size, quarter_size], [quarter_size, quarter_size],
                  [-quarter_size, -quarter_size], [quarter_size, -quarter_size]]
        
        for i, offset in enumerate(offsets):
            new_center = self.center + offset
            self.children[i] = QuadTreeNode(new_center, half_size)
        self.is_leaf = False

    def _insert_particle(self, particle):
        quadrant = self._get_quadrant(particle.position)
        if self.children[quadrant] is None:
            quarter_size = self.size / 4
            offset = [[-quarter_size, quarter_size], [quarter_size, quarter_size],
                     [-quarter_size, -quarter_size], [quarter_size, -quarter_size]][quadrant]
            self.children[quadrant] = QuadTreeNode(self.center + offset, self.size / 2)
        self.children[quadrant].insert(particle)

    def _get_quadrant(self, position):
        if position[0] < self.center[0]:
            return 0 if position[1] > self.center[1] else 2
        else:
            return 1 if position[1] > self.center[1] else 3

def create_bridge(length=10, height=2, num_segments=20):
    particles = []
    dx = length / num_segments
    
    # Create bridge particles
    for i in range(num_segments + 1):
        x = i * dx - length/2
        # Bottom layer
        particles.append(Particle(1.0, [x, 0], [0, 0], fixed=i==0 or i==num_segments))
        # Top layer
        if i < num_segments:
            particles.append(Particle(1.0, [x + dx/2, height], [0, 0]))
    
    # Add connections (springs)
    k_structural = 1000.0  # Spring constant
    k_cross = 500.0      # Cross-support spring constant
    
    for i in range(len(particles)):
        p1 = particles[i]
        for j in range(i + 1, len(particles)):
            p2 = particles[j]
            dist = np.linalg.norm(p1.position - p2.position)
            
            # Connect if within reasonable distance
            if dist < dx * 1.5:
                # Structural connections
                p1.connections.append((p2, dist, k_structural))
                p2.connections.append((p1, dist, k_structural))
            elif dist < dx * 2.0:
                # Cross supports
                p1.connections.append((p2, dist, k_cross))
                p2.connections.append((p1, dist, k_cross))
    
    return particles

def calculate_spring_forces(particles):
    for particle in particles:
        if not particle.fixed:
            # Reset acceleration (gravity)
            particle.acceleration = np.array([0, -9.81])
            
            # Add spring forces
            for connected_particle, rest_length, k in particle.connections:
                direction = connected_particle.position - particle.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    # Spring force
                    force = k * (distance - rest_length) * direction / distance
                    # Damping force
                    damping = 0.5 * (connected_particle.velocity - particle.velocity)
                    total_force = force + damping
                    particle.acceleration += total_force / particle.mass

def calculate_forces_BH(particles, theta=0.5):
    positions = np.array([p.position for p in particles])
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    size = 4 * np.max(distances)
    
    root = QuadTreeNode(center, size)
    for particle in particles:
        root.insert(particle)
    
    calculate_spring_forces(particles)

def calculate_forces_direct(particles):
    calculate_spring_forces(particles)

def update_positions(particles, dt, use_BH=True):
    # Calculate forces
    if use_BH:
        calculate_forces_BH(particles)
    else:
        calculate_forces_direct(particles)
    
    # Update positions using Verlet integration
    for particle in particles:
        if not particle.fixed:
            particle.position += particle.velocity * dt + 0.5 * particle.acceleration * dt**2
            old_acceleration = particle.acceleration.copy()
            
            # Recalculate forces with new positions
            if use_BH:
                calculate_forces_BH(particles)
            else:
                calculate_forces_direct(particles)
            
            particle.velocity += 0.5 * (old_acceleration + particle.acceleration) * dt
        particle.history.append(particle.position.copy())

# Create simulation
particles = create_bridge(length=10, height=2, num_segments=20)

# Simulation parameters
dt = 0.001
num_steps = 1000

# Add initial perturbation to create collapse
middle_particle = particles[len(particles)//2]
middle_particle.velocity = np.array([0, -0.1])

# Run simulation
for step in range(num_steps):
    update_positions(particles, dt, use_BH=True)

# Create animation
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(-6, 6)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Bridge Collapse Simulation')

# Create line collections for springs
lines = []
spring_lines = LineCollection([], colors='gray', alpha=0.5)
ax.add_collection(spring_lines)

# Create points for particles
points = [ax.plot([], [], 'o')[0] for _ in particles]
for point, particle in zip(points, particles):
    point.set_color('red' if particle.fixed else 'blue')
    point.set_markersize(8 if particle.fixed else 6)

def animate(frame):
    # Update spring connections
    segments = []
    for particle in particles:
        for connected_particle, _, _ in particle.connections:
            segments.append([particle.history[frame], connected_particle.history[frame]])
    spring_lines.set_segments(segments)
    
    # Update particle positions
    for point, particle in zip(points, particles):
        point.set_data([particle.history[frame][0]], [particle.history[frame][1]])
    
    return points + [spring_lines]

anim = FuncAnimation(fig, animate, frames=num_steps, 
                    interval=20, blit=True)

from matplotlib.animation import PillowWriter, FFMpegWriter

# Option 1: Save as GIF
writer_gif = PillowWriter(fps=30)
anim.save('bridge_collapse.gif', writer=writer_gif)

plt.show()