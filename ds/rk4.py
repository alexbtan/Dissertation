import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Class to define each body of mass
class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2)
        # Used to simulate
        self.history = [self.position.copy()]

def calculate_acceleration(bodies):
    for body in bodies:
        body.acceleration.fill(0)
        
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i != j:
                r = body2.position - body1.position
                r_mag = np.linalg.norm(r) # magnitude of distance
                body1.acceleration += G * body2.mass * r / (r_mag ** 3)
    
    return np.array([body.acceleration for body in bodies])

# RK4 step function
def update_positions(bodies, dt):
    velocities = np.array([body.velocity for body in bodies])

    # Compute k1
    k1_v = calculate_acceleration(bodies) * dt
    k1_r = velocities * dt

    # Compute k2
    temp_bodies = [Body(b.mass, b.position + k1_r[i] / 2, b.velocity + k1_v[i] / 2) for i, b in enumerate(bodies)]
    k2_v = calculate_acceleration(temp_bodies) * dt
    k2_r = (velocities + k1_v / 2) * dt

    # Compute k3
    temp_bodies = [Body(b.mass, b.position + k2_r[i] / 2, b.velocity + k2_v[i] / 2) for i, b in enumerate(bodies)]
    k3_v = calculate_acceleration(temp_bodies) * dt
    k3_r = (velocities + k2_v / 2) * dt

    # Compute k4
    temp_bodies = [Body(b.mass, b.position + k3_r[i], b.velocity + k3_v[i]) for i, b in enumerate(bodies)]
    k4_v = calculate_acceleration(temp_bodies) * dt
    k4_r = (velocities + k3_v) * dt

    # Update positions and velocities
    for i, body in enumerate(bodies):
        body.position += (k1_r[i] + 2 * k2_r[i] + 2 * k3_r[i] + k4_r[i]) / 6
        body.velocity += (k1_v[i] + 2 * k2_v[i] + 2 * k3_v[i] + k4_v[i]) / 6
        body.history.append(body.position.copy())

G=1.0
L = 1.0  # Side length of the equilateral triangle
omega = np.sqrt(G / L**3)  # Angular velocity for stability

bodies = [
    Body(1.0, [L / 2, np.sqrt(3) * L / 6], [-omega * np.sqrt(3) * L / 6, omega * L / 2]),
    Body(1.0, [-L / 2, np.sqrt(3) * L / 6], [-omega * np.sqrt(3) * L / 6, -omega * L / 2]),
    Body(1.0, [0, -np.sqrt(3) * L / 3], [omega * np.sqrt(3) * L / 3, 0])
]

# Simulation parameters
dt = 0.001
num_steps = 10000

# Run simulation
for step in range(num_steps):
    update_positions(bodies, dt)

# Create animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Three-Body Figure-8 Orbit')

lines = [ax.plot([], [], '-', alpha=0.5)[0] for _ in bodies]
points = [ax.plot([], [], 'o')[0] for _ in bodies]
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

plt.show()