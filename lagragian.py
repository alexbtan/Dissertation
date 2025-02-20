import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Body:
    def __init__(self, mass, position, momentum):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.momentum = np.array(momentum, dtype=float)  # Using momentum instead of velocity
        self.history = [self.position.copy()]

def compute_kinetic_energy(bodies):
    """T = Σᵢ |pᵢ|²/(2mᵢ)"""
    T = 0
    for body in bodies:
        T += np.sum(body.momentum**2) / (2 * body.mass)
    return T

def compute_potential_energy(bodies, G=1.0, epsilon=0.1):
    """V = -G Σᵢ Σⱼ₍ⱼ₎ᵢ mᵢmⱼ/√(|rᵢ - rⱼ|² + ε²)"""
    V = 0
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies[i+1:], i+1):
            r = body1.position - body2.position
            r_mag = np.sqrt(np.sum(r**2) + epsilon**2)
            V -= G * body1.mass * body2.mass / r_mag
    return V

def compute_forces(bodies, G=1.0, epsilon=0.1):
    """∂L/∂r = -∂V/∂r"""
    forces = [np.zeros(2) for _ in bodies]
    
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i != j:
                r = body1.position - body2.position
                r_mag = np.sqrt(np.sum(r**2) + epsilon**2)
                # Force is negative gradient of potential
                forces[i] += G * body1.mass * body2.mass * r / (r_mag**3)
    
    return forces

def symplectic_step(bodies, dt, G=1.0):
    """Symplectic integration preserving geometric structure"""
    # Half step in momentum (p₁/₂ = p₀ + F(q₀)dt/2)
    forces = compute_forces(bodies, G)
    for body, force in zip(bodies, forces):
        body.momentum += 0.5 * dt * force
    
    # Full step in position (q₁ = q₀ + (p₁/₂/m)dt)
    for body in bodies:
        body.position += dt * body.momentum / body.mass
        body.history.append(body.position.copy())
    
    # Half step in momentum (p₁ = p₁/₂ + F(q₁)dt/2)
    forces = compute_forces(bodies, G)
    for body, force in zip(bodies, forces):
        body.momentum += 0.5 * dt * force

# Initialize bodies (e.g., figure-8 orbit)
bodies = [
    Body(1.0, [0.97000436, -0.24308753], [-0.93240737/2 * 1.0, -0.86473146/2 * 1.0]),
    Body(1.0, [-0.97000436, 0.24308753], [-0.93240737/2 * 1.0, -0.86473146/2 * 1.0]),
    Body(1.0, [0, 0], [0.93240737 * 1.0, 0.86473146 * 1.0])
]

# Simulation parameters
dt = 0.001
num_steps = 10000

# Track energy
total_energy = []

# Run simulation
for step in range(num_steps):
    T = compute_kinetic_energy(bodies)
    V = compute_potential_energy(bodies)
    total_energy.append(T + V)
    symplectic_step(bodies, dt)

# Create animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title('Three-Body Orbit (Lagrangian Formulation)')

lines = [ax1.plot([], [], '-', alpha=0.5)[0] for _ in bodies]
points = [ax1.plot([], [], 'o')[0] for _ in bodies]
colors = ['red', 'blue', 'green']

for line, point, color in zip(lines, points, colors):
    line.set_color(color)
    point.set_color(color)

# Plot energy conservation
ax2.plot(total_energy)
ax2.set_title('Total Energy vs Time')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Total Energy')
ax2.grid(True)

def animate(frame):
    for i, (body, line, point) in enumerate(zip(bodies, lines, points)):
        history = np.array(body.history[:frame+1])
        line.set_data(history[:, 0], history[:, 1])
        point.set_data([body.history[frame][0]], [body.history[frame][1]])
    return lines + points

anim = FuncAnimation(fig, animate, frames=num_steps, 
                    interval=0.5, blit=True)

plt.show()