import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        # Convert numpy arrays to PyTorch tensors on GPU
        self.position = torch.tensor(position, dtype=torch.float32, device=device)
        self.velocity = torch.tensor(velocity, dtype=torch.float32, device=device)
        self.acceleration = torch.zeros(2, dtype=torch.float32, device=device)
        self.history = []
        self.history_length = 50

class QuadTreeNode:
    def __init__(self, center, size):
        self.center = torch.tensor(center, dtype=torch.float32, device=device)
        self.size = size
        self.mass = 0.0
        self.com = torch.zeros(2, dtype=torch.float32, device=device)
        self.children = [None] * 4
        self.body = None
        self.is_leaf = True

    def insert(self, body):
        if self.body is None and all(child is None for child in self.children):
            self.body = body
            self.mass = body.mass
            self.com = body.position.clone()
            return
        
        if self.body is not None:
            self._split()
            self._insert_body(self.body)
            self.body = None
            self._insert_body(body)
        else:
            self._insert_body(body)
        
        # Update mass and center of mass
        self.mass = sum(child.mass for child in self.children if child is not None)
        self.com = sum(child.com * child.mass for child in self.children if child is not None) / self.mass

    def _split(self):
        quarter_size = self.size / 4
        half_size = self.size / 2
        
        offsets = torch.tensor([
            [-quarter_size, quarter_size],
            [quarter_size, quarter_size],
            [-quarter_size, -quarter_size],
            [quarter_size, -quarter_size]
        ], dtype=torch.float32, device=device)
        
        for i, offset in enumerate(offsets):
            new_center = self.center + offset
            self.children[i] = QuadTreeNode(new_center, half_size)
        
        self.is_leaf = False

    def _insert_body(self, body):
        quadrant = self._get_quadrant(body.position)
        if self.children[quadrant] is None:
            quarter_size = self.size / 4
            offsets = torch.tensor([
                [-quarter_size, quarter_size],
                [quarter_size, quarter_size],
                [-quarter_size, -quarter_size],
                [quarter_size, -quarter_size]
            ], dtype=torch.float32, device=device)
            self.children[quadrant] = QuadTreeNode(self.center + offsets[quadrant], self.size / 2)
        self.children[quadrant].insert(body)

    def _get_quadrant(self, position):
        if position[0] < self.center[0]:
            return 0 if position[1] > self.center[1] else 2
        else:
            return 1 if position[1] > self.center[1] else 3

def calculate_force_BH(body, node, theta=0.5, G=1.0, epsilon=0.1):
    r = node.com - body.position
    r_mag = torch.norm(r)
    
    if r_mag == 0:
        return torch.zeros(2, dtype=torch.float32, device=device)
    
    if node.is_leaf or (node.size / r_mag < theta):
        return G * node.mass * r / ((r_mag**2 + epsilon**2) ** (3/2))
    
    acceleration = torch.zeros(2, dtype=torch.float32, device=device)
    for child in node.children:
        if child is not None and child.mass > 0:
            acceleration += calculate_force_BH(body, child, theta, G, epsilon)
    return acceleration

def calculate_acceleration_BH(bodies, theta=0.5):
    # Find bounds for quadtree
    positions = torch.stack([body.position for body in bodies])
    center = torch.mean(positions, dim=0)
    
    # Calculate maximum distance from center
    distances_from_center = torch.norm(positions - center, dim=1)
    size = 4 * torch.max(distances_from_center)
    
    # Build quadtree
    root = QuadTreeNode(center, size)
    for body in bodies:
        root.insert(body)
    
    # Calculate accelerations
    for body in bodies:
        body.acceleration = calculate_force_BH(body, root, theta)

def update_positions(bodies, dt):
    for body in bodies:
        # Velocity Verlet integration
        body.position += body.velocity * dt + 0.5 * body.acceleration * dt**2
        old_acceleration = body.acceleration.clone()
        
        calculate_acceleration_BH(bodies)
        
        body.velocity += 0.5 * (old_acceleration + body.acceleration) * dt
        
        # Update history (convert to CPU for matplotlib)
        body.history.append(body.position.cpu().numpy())
        if len(body.history) > body.history_length:
            body.history.pop(0)

class Simulation:
    def __init__(self, num_bodies=100):
        self.bodies = []
        self.dt = 0.01
        self.create_random_bodies(num_bodies)
        
    def create_random_bodies(self, n):
        for _ in range(n):
            position = (np.random.rand(2) - 0.5) * 2
            velocity = (np.random.rand(2) - 0.5) * 0.5
            mass = np.random.uniform(0.1, 2.0)
            self.bodies.append(Body(mass, position, velocity))
    
    def step(self):
        update_positions(self.bodies, self.dt)

# Create figure and simulation
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
sim = Simulation(num_bodies=20)  # Can handle more bodies now!

# Create scatter plot
scatter = ax.scatter([b.position.cpu().numpy()[0] for b in sim.bodies],
                    [b.position.cpu().numpy()[1] for b in sim.bodies],
                    c='blue', s=[b.mass * 20 for b in sim.bodies])

# Create line objects for trails
lines = [ax.plot([], [], '-', alpha=0.3)[0] for _ in sim.bodies]

def update(frame):
    # Perform simulation step
    sim.step()
    
    # Update scatter positions (convert from GPU to CPU)
    scatter.set_offsets([b.position.cpu().numpy() for b in sim.bodies])
    
    # Update trails
    for line, body in zip(lines, sim.bodies):
        if body.history:
            history = np.array(body.history)
            line.set_data(history[:, 0], history[:, 1])
    
    return [scatter] + lines

# Create animation
anim = FuncAnimation(fig, update, frames=None,
                    interval=20, blit=True,
                    cache_frame_data=False)  # Add this parameter

# Add this to prevent garbage collection
plt.rcParams['animation.html'] = 'html5'

plt.show()