import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import signal
import sys
import time

# Handle keyboard interrupts
def signal_handler(sig, frame):
    print('\nExiting simulation gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros(2)
        self.history = [self.position.copy()]

def calculate_acceleration(bodies, G=1.0):
    # Compute accelerations using vectorized NumPy operations
    n = len(bodies)
    
    # Reset accelerations
    for body in bodies:
        body.acceleration = np.zeros(2)
    
    # Compute gravitational acceleration
    for i in range(n):
        for j in range(n):
            if i != j:
                # Vector from body i to body j
                r = bodies[j].position - bodies[i].position
                
                # Distance between bodies
                r_mag = np.linalg.norm(r)
                
                # Gravitational force magnitude
                force_mag = G * bodies[i].mass * bodies[j].mass / (r_mag ** 2)
                
                # Force direction (normalized vector)
                force_dir = r / r_mag
                
                # Acceleration for body i
                bodies[i].acceleration += force_dir * force_mag / bodies[i].mass

def update_positions(bodies, dt):
    # First compute all accelerations
    calculate_acceleration(bodies)
    
    for body in bodies:
        # Verlet integration
        # Update position
        body.position += body.velocity * dt + 0.5 * body.acceleration * dt**2
        
        # Store old acceleration
        old_acceleration = body.acceleration.copy()
        
        # Recompute accelerations
        calculate_acceleration(bodies)
        
        # Update velocity with average acceleration
        body.velocity += 0.5 * (old_acceleration + body.acceleration) * dt
        
        # Store position history
        body.history.append(body.position.copy())

def run_simulation():
    try:
        # Initialize bodies with more precise initial conditions for figure-8 orbit
        bodies = [
            Body(1.0, [0.97000436, -0.24308753], [-0.93240737/2, -0.86473146/2]),
            Body(1.0, [-0.97000436, 0.24308753], [-0.93240737/2, -0.86473146/2]),
            Body(1.0, [0, 0], [0.93240737, 0.86473146])
        ]

        # Simulation parameters
        dt = 0.01  # Larger time step for real-time simulation
        real_time_factor = 0.05  # Slow down the simulation 

        # Matplotlib setup
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Three-Body Figure-8 Orbit')

        # Setup plot elements
        lines = [ax.plot([], [], '-', alpha=0.5)[0] for _ in bodies]
        points = [ax.plot([], [], 'o')[0] for _ in bodies]
        colors = ['red', 'blue', 'green']

        for line, point, color in zip(lines, points, colors):
            line.set_color(color)
            point.set_color(color)

        # Simulation loop
        start_time = time.time()
        step = 0

        while True:
            # Update positions
            update_positions(bodies, dt)
            
            # Clear previous plot
            ax.clear()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.grid(True)
            ax.set_title('Three-Body Figure-8 Orbit')

            # Replot lines and points
            for body, color in zip(bodies, colors):
                history = np.array(body.history)
                ax.plot(history[:, 0], history[:, 1], '-', color=color, alpha=0.5)
                ax.plot([body.position[0]], [body.position[1]], 'o', color=color)

            # Refresh plot
            plt.pause(real_time_factor)
            plt.draw()

            # Update step counter
            step += 1
            if step % 10 == 0:
                print(f"Step {step}, Elapsed Time: {time.time() - start_time:.2f}s", end='\r')

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    run_simulation()