import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        # Store momentum instead of velocity for Hamiltonian mechanics
        self.momentum = mass * np.array(velocity, dtype=np.float64)
        self.force = np.zeros(2, dtype=np.float64)
        self.history = [self.position.copy()]

class QuadTreeNode:
    def __init__(self, center, size):
        self.center = np.array(center, dtype=np.float64)
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(2, dtype=np.float64)
        self.momentum = np.zeros(2, dtype=np.float64)  # Track momentum for Hamiltonian
        self.children = [None] * 4
        self.body = None
        self.is_leaf = True

    def insert(self, body):
        if self.body is None and all(child is None for child in self.children):
            self.body = body
            self.mass = body.mass
            self.com = body.position.copy()
            self.momentum = body.momentum.copy()
            return
        
        if self.body is not None:
            self._split()
            self._insert_body(self.body)
            self.body = None
            self._insert_body(body)
        else:
            self._insert_body(body)
        
        # Update mass, center of mass, and momentum
        self.mass = sum(child.mass for child in self.children if child is not None)
        self.com = sum(child.com * child.mass for child in self.children if child is not None) / self.mass
        self.momentum = sum(child.momentum for child in self.children if child is not None)

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
                [-quarter_size, quarter_size],
                [quarter_size, quarter_size],
                [-quarter_size, -quarter_size],
                [quarter_size, -quarter_size]
            ][quadrant]
            self.children[quadrant] = QuadTreeNode(self.center + offset, self.size / 2)
        self.children[quadrant].insert(body)

    def _get_quadrant(self, position):
        if position[0] < self.center[0]:
            return 0 if position[1] > self.center[1] else 2
        else:
            return 1 if position[1] > self.center[1] else 3

def calculate_force_BH_hamiltonian(body, node, theta=0.5, G=1.0, epsilon=0.1):
    """Calculate force using Hamiltonian formulation with Barnes-Hut"""
    r = node.com - body.position
    r_mag = np.linalg.norm(r)
    
    if r_mag == 0:
        return np.zeros(2, dtype=np.float64)
    
    # Use Barnes-Hut criterion for approximation
    if node.is_leaf or (node.size / r_mag < theta):
        # Force from potential energy gradient
        return G * node.mass * r / ((r_mag**2 + epsilon**2) ** (3/2))
    
    # Recursively calculate forces from children
    force = np.zeros(2, dtype=np.float64)
    for child in node.children:
        if child is not None and child.mass > 0:
            force += calculate_force_BH_hamiltonian(body, child, theta, G, epsilon)
    return force

def symplectic_update(bodies, dt, theta=0.5):
    """Symplectic integration with Barnes-Hut approximation"""
    # Build tree
    positions = np.array([body.position for body in bodies])
    center = np.mean(positions, axis=0)
    size = 4 * np.max(np.linalg.norm(positions - center, axis=1))
    root = QuadTreeNode(center, size)
    
    for body in bodies:
        root.insert(body)
    
    # First half-step in momentum (p₁/₂ = p₀ + F(q₀)dt/2)
    for body in bodies:
        force = calculate_force_BH_hamiltonian(body, root, theta)
        body.momentum += 0.5 * dt * force
    
    # Full step in position (q₁ = q₀ + (p₁/₂/m)dt)
    for body in bodies:
        body.position += dt * body.momentum / body.mass
        body.history.append(body.position.copy())
    
    # Rebuild tree with new positions
    root = QuadTreeNode(center, size)
    for body in bodies:
        root.insert(body)
    
    # Second half-step in momentum (p₁ = p₁/₂ + F(q₁)dt/2)
    for body in bodies:
        force = calculate_force_BH_hamiltonian(body, root, theta)
        body.momentum += 0.5 * dt * force

def calculate_total_energy(bodies, root, G=1.0):
    """Calculate total energy of the system"""
    kinetic = sum(np.sum(body.momentum**2)/(2*body.mass) for body in bodies)
    potential = 0
    
    for body in bodies:
        force = calculate_force_BH_hamiltonian(body, root, G=G)
        potential -= 0.5 * G * body.mass * np.sum(force * body.position)
    
    return kinetic + potential

# Create simulation
bodies = [
    Body(1.0, [0.97000436, -0.24308753], [-0.93240737/2, -0.86473146/2]),
    Body(1.0, [-0.97000436, 0.24308753], [-0.93240737/2, -0.86473146/2]),
    Body(1.0, [0, 0], [0.93240737, 0.86473146])
]

# Simulation parameters
dt = 0.01
num_steps = 1000
energy_history = []

# Run simulation
for step in range(num_steps):
    symplectic_update(bodies, dt)
    
    # Calculate energy (optional, for monitoring)
    positions = np.array([body.position for body in bodies])
    center = np.mean(positions, axis=0)
    size = 4 * np.max(np.linalg.norm(positions - center, axis=1))
    root = QuadTreeNode(center, size)
    for body in bodies:
        root.insert(body)
    energy_history.append(calculate_total_energy(bodies, root))

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot trajectories
for body in bodies:
    history = np.array(body.history)
    ax1.plot(history[:, 0], history[:, 1])
ax1.set_title('Trajectories')
ax1.set_aspect('equal')
ax1.grid(True)

# Plot energy conservation
ax2.plot(energy_history)
ax2.set_title('Total Energy')
ax2.grid(True)

plt.show()