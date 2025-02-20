import numpy as np
import matplotlib.pyplot as plt

class NewtonianNBody:
    def __init__(self, masses, positions, velocities):
        # Ensure everything is float64
        self.masses = np.array(masses, dtype=np.float64)
        self.positions = np.array(positions, dtype=np.float64)
        self.velocities = np.array(velocities, dtype=np.float64)
        self.forces = np.zeros_like(self.positions, dtype=np.float64)
        self.energy_history = []
        
    def calculate_forces(self):
        self.forces.fill(0)
        G = 1.0
        
        for i in range(len(self.masses)):
            for j in range(len(self.masses)):
                if i != j:
                    r = self.positions[j] - self.positions[i]
                    r_mag = np.linalg.norm(r)
                    self.forces[i] += G * self.masses[i] * self.masses[j] * r / r_mag**3
    
    def step(self, dt):
        self.positions += self.velocities * dt + 0.5 * self.forces/self.masses[:, np.newaxis] * dt**2
        old_forces = self.forces.copy()
        
        self.calculate_forces()
        
        self.velocities += 0.5 * (old_forces + self.forces)/self.masses[:, np.newaxis] * dt
        self.energy_history.append(self.total_energy())
    
    def total_energy(self):
        kinetic = 0.5 * np.sum(self.masses * np.sum(self.velocities**2, axis=1))
        potential = 0
        G = 1.0
        for i in range(len(self.masses)):
            for j in range(i+1, len(self.masses)):
                r = np.linalg.norm(self.positions[i] - self.positions[j])
                potential -= G * self.masses[i] * self.masses[j] / r
        return kinetic + potential

class HamiltonianNBody:
    def __init__(self, masses, positions, velocities):
        self.masses = np.array(masses, dtype=np.float64)
        self.positions = np.array(positions, dtype=np.float64)
        velocities = np.array(velocities, dtype=np.float64)
        self.momenta = self.masses[:, np.newaxis] * velocities
        self.forces = np.zeros_like(self.positions, dtype=np.float64)
        self.energy_history = []
    
    def calculate_forces(self):
        self.forces.fill(0)
        G = 1.0
        
        for i in range(len(self.masses)):
            for j in range(len(self.masses)):
                if i != j:
                    r = self.positions[j] - self.positions[i]
                    r_mag = np.linalg.norm(r)
                    self.forces[i] += G * self.masses[i] * self.masses[j] * r / r_mag**3
    
    def step(self, dt):
        self.calculate_forces()
        self.momenta += 0.5 * dt * self.forces
        
        self.positions += dt * self.momenta/self.masses[:, np.newaxis]
        
        self.calculate_forces()
        self.momenta += 0.5 * dt * self.forces
        
        self.energy_history.append(self.total_energy())
    
    def total_energy(self):
        kinetic = np.sum(self.momenta**2/(2*self.masses[:, np.newaxis]))
        potential = 0
        G = 1.0
        for i in range(len(self.masses)):
            for j in range(i+1, len(self.masses)):
                r = np.linalg.norm(self.positions[i] - self.positions[j])
                potential -= G * self.masses[i] * self.masses[j] / r
        return kinetic + potential

def run_comparison():
    # Initial conditions (ensure all numbers are floats)
    masses = [1.0, 1.0, 1.0]
    positions = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    velocities = [[0.0, 0.5], [-0.5, 0.0], [0.5, 0.0]]
    
    newtonian = NewtonianNBody(masses, positions, velocities)
    hamiltonian = HamiltonianNBody(masses, positions, velocities)
    
    steps = 1000
    dt = 0.01
    
    for _ in range(steps):
        newtonian.step(dt)
        hamiltonian.step(dt)
    
    plt.figure(figsize=(12,6))
    
    plt.subplot(121)
    plt.title('Trajectories')
    for i in range(3):
        plt.plot(newtonian.positions[i,0], newtonian.positions[i,1], 'r.', label=f'Newtonian {i+1}')
        plt.plot(hamiltonian.positions[i,0], hamiltonian.positions[i,1], 'b.', label=f'Hamiltonian {i+1}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    plt.title('Energy Conservation')
    plt.plot(newtonian.energy_history, 'r-', label='Newtonian')
    plt.plot(hamiltonian.energy_history, 'b-', label='Hamiltonian')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

run_comparison()