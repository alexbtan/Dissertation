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

def calculate_total_energy(bodies, G=1.0):
    # Kinetic Energy: KE = 1/2 * m * v^2
    kinetic = sum(0.5 * body.mass * np.sum(body.velocity**2) for body in bodies)
    
    # Potential Energy: PE = -G * m1 * m2 / r
    potential = 0
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies[i+1:], i+1):
            r = np.linalg.norm(body2.position - body1.position)
            potential -= G * body1.mass * body2.mass / r
    
    return kinetic + potential

def calculate_acceleration(bodies, G=1.0):
    for body in bodies:
        body.acceleration.fill(0)
        
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i != j:
                r = body2.position - body1.position
                r_mag = np.linalg.norm(r)
                body1.acceleration += G * body2.mass * r / (r_mag ** 3)

def update_positions(bodies, dt):
    for body in bodies:
        # Velocity Verlet integration
        body.position += body.velocity * dt + 0.5 * body.acceleration * dt**2
        old_acceleration = body.acceleration.copy()
        
        # Recalculate accelerations with new positions
        calculate_acceleration(bodies)
        
        # Update velocities using average of old and new accelerations
        # This works 
        body.velocity += 0.5 * (old_acceleration + body.acceleration) * dt
        body.history.append(body.position.copy())


# Initialize bodies
bodies = [
    Body(1.0, [0.97000436, -0.24308753], [-0.93240737/2, -0.86473146/2]),
    Body(1.0, [-0.97000436, 0.24308753], [-0.93240737/2, -0.86473146/2]),
    Body(1.0, [0, 0], [0.93240737, 0.86473146])
]

# Simulation parameters
dt = 0.001
num_steps = 10000

# Track energy
energy_history = []

# Run simulation
for step in range(num_steps):
    update_positions(bodies, dt)
    energy_history.append(calculate_total_energy(bodies))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# Orbit plot
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title('Three-Body Figure-8 Orbit')

# Energy plot
ax2.plot(energy_history, label='Total Energy')
ax2.set_title('Total Energy over Time')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Energy')
ax2.grid(True)
ax2.legend()

# Setup animation
lines = [ax1.plot([], [], '-', alpha=0.5)[0] for _ in bodies]
points = [ax1.plot([], [], 'o')[0] for _ in bodies]
colors = ['red', 'blue', 'green']

for line, point, color in zip(lines, points, colors):
    line.set_color(color)
    point.set_color(color)

def animate(frame):
    for i, (body, line, point) in enumerate(zip(bodies, lines, points)):
        history = np.array(body.history[:frame+1])
        line.set_data(history[:, 0], history[:, 1])
        point.set_data([body.history[frame][0]], [body.history[frame][1]])
    return lines + points

anim = FuncAnimation(fig, animate, frames=num_steps, 
                    interval=0.005, blit=True)

plt.tight_layout()
plt.show()