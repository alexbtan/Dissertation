import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2)
        self.history = [self.position.copy()]

class QuadTreeNode:
    def __init__(self, center, size):
        self.center = np.array(center)
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(2)  # Center of mass
        self.children = [None] * 4  # NW, NE, SW, SE
        self.body = None
        self.is_leaf = True

    def insert(self, body):
        if self.body is None and all(child is None for child in self.children):
            self.body = body
            self.mass = body.mass
            self.com = body.position.copy()
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
        
        offsets = [
            [-quarter_size, quarter_size],  # NW
            [quarter_size, quarter_size],   # NE
            [-quarter_size, -quarter_size], # SW
            [quarter_size, -quarter_size]   # SE
        ]
        
        for i, offset in enumerate(offsets):
            new_center = self.center + offset
            self.children[i] = QuadTreeNode(new_center, half_size)
        
        self.is_leaf = False

    def _insert_body(self, body):
        quadrant = self._get_quadrant(body.position)
        if self.children[quadrant] is None:
            quarter_size = self.size / 4
            offset = [
                [-quarter_size, quarter_size],  # NW
                [quarter_size, quarter_size],   # NE
                [-quarter_size, -quarter_size], # SW
                [quarter_size, -quarter_size]   # SE
            ][quadrant]
            self.children[quadrant] = QuadTreeNode(self.center + offset, self.size / 2)
        self.children[quadrant].insert(body)

    def _get_quadrant(self, position):
        if position[0] < self.center[0]:
            return 0 if position[1] > self.center[1] else 2  # NW or SW
        else:
            return 1 if position[1] > self.center[1] else 3  # NE or SE

def calculate_force_BH(body, node, theta=0.5, G=1.0, epsilon=0.1):  # Added epsilon parameter
    r = node.com - body.position
    r_mag = np.linalg.norm(r)
    
    if r_mag == 0:  # Same body
        return np.zeros(2)
    
    if node.is_leaf or (node.size / r_mag < theta):
        # Use center of mass approximation with softening
        return G * node.mass * r / ((r_mag**2 + epsilon**2) ** (3/2))
    
    # Otherwise, recurse on children
    acceleration = np.zeros(2)
    for child in node.children:
        if child is not None and child.mass > 0:
            acceleration += calculate_force_BH(body, child, theta, G, epsilon)
    return acceleration

def calculate_acceleration_BH(bodies, theta=0.5):
    # Find bounds for quadtree
    positions = np.array([body.position for body in bodies])
    center = np.mean(positions, axis=0)
    
    # Fix: Calculate maximum distance from center correctly
    distances_from_center = np.linalg.norm(positions - center, axis=1)
    size = 4 * np.max(distances_from_center)  # Make the box twice as big as needed to handle edge cases
    
    # Build quadtree
    root = QuadTreeNode(center, size)
    for body in bodies:
        root.insert(body)
    
    # Calculate accelerations
    for body in bodies:
        body.acceleration = calculate_force_BH(body, root, theta)

def calculate_acceleration_direct(bodies, G=1.0):
    for body in bodies:
        body.acceleration.fill(0)
        
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i != j:
                r = body2.position - body1.position
                r_mag = np.linalg.norm(r)
                body1.acceleration += G * body2.mass * r / ((r_mag + 1e-6) ** 3)

def update_positions(bodies, dt, use_BH=True):
    for body in bodies:
        body.position += body.velocity * dt + 0.5 * body.acceleration * dt**2
        old_acceleration = body.acceleration.copy()
        
        if use_BH:
            calculate_acceleration_BH(bodies)
        else:
            calculate_acceleration_direct(bodies)
        
        body.velocity += 0.5 * (old_acceleration + body.acceleration) * dt
        body.history.append(body.position.copy())

def create_random_bodies(N, space_size=4.0, max_velocity=0.5, mass_range=(0.1, 2.0)):
    bodies = []
    for _ in range(N):
        position = (np.random.rand(2) - 0.5) * space_size
        velocity = (np.random.rand(2) - 0.5) * max_velocity
        mass = np.random.uniform(mass_range[0], mass_range[1])
        bodies.append(Body(mass, position, velocity))
    return bodies

# Simulation parameters
N = 50  # Increased number of bodies to show Barnes-Hut advantage
dt = 0.01
num_steps = 1000

# Create bodies
bodies = create_random_bodies(N)

# Run simulation using Barnes-Hut
for step in range(num_steps):
    update_positions(bodies, dt, use_BH=True)

# Create animation
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title(f'{N}-Body Barnes-Hut Simulation (θ=0.5)')

lines = [ax.plot([], [], '-', alpha=0.3)[0] for _ in bodies]
points = [ax.plot([], [], 'o')[0] for _ in bodies]

colors = plt.cm.viridis(np.linspace(0, 1, len(bodies)))
sizes = [50 * body.mass for body in bodies]

for i, (line, point, color) in enumerate(zip(lines, points, colors)):
    line.set_color(color)
    point.set_color(color)
    point.set_markersize(np.sqrt(sizes[i]))

def animate(frame):
    for i, (body, line, point) in enumerate(zip(bodies, lines, points)):
        history = np.array(body.history[:frame+1])
        line.set_data(history[:, 0], history[:, 1])
        point.set_data([body.history[frame][0]], [body.history[frame][1]])
    return lines + points

anim = FuncAnimation(fig, animate, frames=num_steps, 
                    interval=1, blit=True)

plt.show()