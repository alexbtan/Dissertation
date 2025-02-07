import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

class Person:
    def __init__(self, position, velocity, status='S'):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.status = status  # S: Susceptible, I: Infected, R: Recovered
        self.infection_time = None
        self.social_distance = np.random.uniform(0.5, 1.5)  # Individual compliance
        
class QuadTreeNode:
    def __init__(self, center, size):
        self.center = np.array(center)
        self.size = size
        self.people = []
        self.children = [None] * 4
        self.total_count = 0
        self.infected_count = 0
    
    def insert(self, person):
        if len(self.people) < 4 and all(c is None for c in self.children):
            self.people.append(person)
            self.total_count += 1
            if person.status == 'I':
                self.infected_count += 1
            return
        
        if not self.children[0]:
            self._split()
        
        quadrant = self._get_quadrant(person.position)
        self.children[quadrant].insert(person)
        self.total_count += 1
        if person.status == 'I':
            self.infected_count += 1
    
    def _split(self):
        s = self.size / 4
        for i, offset in enumerate([
            [-s, s], [s, s],      # NW, NE
            [-s, -s], [s, -s]     # SW, SE
        ]):
            self.children[i] = QuadTreeNode(self.center + offset, self.size/2)
        
        for person in self.people:
            quadrant = self._get_quadrant(person.position)
            self.children[quadrant].insert(person)
        self.people = []
    
    def _get_quadrant(self, position):
        if position[0] < self.center[0]:
            return 0 if position[1] > self.center[1] else 2
        else:
            return 1 if position[1] > self.center[1] else 3

    def query_range(self, position, radius):
        """Find all people within radius of position"""
        if not self._intersects_circle(position, radius):
            return []
        
        found = []
        if self.people:
            for person in self.people:
                if np.linalg.norm(person.position - position) <= radius:
                    found.append(person)
            return found
        
        for child in self.children:
            if child:
                found.extend(child.query_range(position, radius))
        return found
    
    def _intersects_circle(self, center, radius):
        """Check if node's box intersects with circle"""
        dx = abs(center[0] - self.center[0])
        dy = abs(center[1] - self.center[1])
        
        if dx > (self.size + radius) or dy > (self.size + radius):
            return False
        
        if dx <= self.size or dy <= self.size:
            return True
        
        corner_dist = (dx - self.size)**2 + (dy - self.size)**2
        return corner_dist <= radius**2

class DiseaseSimulation:
    def __init__(self, num_people=200, space_size=100):
        self.space_size = space_size
        self.people = []
        self.transmission_rate = 0.3
        self.recovery_time = 200  # time steps
        self.infection_radius = 2.0
        self.time = 0
        self.stats_history = {'S': [], 'I': [], 'R': []}
        
        # Initialize population
        for _ in range(num_people):
            pos = np.random.uniform(0, space_size, 2)
            vel = np.random.normal(0, 0.5, 2)
            status = 'I' if np.random.random() < 0.05 else 'S'
            person = Person(pos, vel, status)
            if status == 'I':
                person.infection_time = 0
            self.people.append(person)
    
    def build_tree(self):
        root = QuadTreeNode([self.space_size/2]*2, self.space_size)
        for person in self.people:
            root.insert(person)
        return root
    
    def update(self):
        self.time += 1
        tree = self.build_tree()
        
        # Update positions and handle collisions
        for person in self.people:
            # Update position
            person.position += person.velocity
            
            # Bounce off walls
            for i in range(2):
                if person.position[i] < 0:
                    person.position[i] = 0
                    person.velocity[i] *= -1
                elif person.position[i] > self.space_size:
                    person.position[i] = self.space_size
                    person.velocity[i] *= -1
            
            # Social distancing forces
            nearby = tree.query_range(person.position, 5.0)
            for other in nearby:
                if other is not person:
                    diff = person.position - other.position
                    dist = np.linalg.norm(diff)
                    if dist < 3.0:
                        # Apply repulsion force
                        force = diff * 0.1 * person.social_distance / (dist + 1e-6)
                        person.velocity += force
            
            # Limit velocity
            speed = np.linalg.norm(person.velocity)
            if speed > 2.0:
                person.velocity *= 2.0 / speed
        
        # Disease transmission and recovery
        for person in self.people:
            if person.status == 'I':
                # Check for recovery
                if self.time - person.infection_time >= self.recovery_time:
                    person.status = 'R'
                else:
                    # Attempt transmission
                    nearby = tree.query_range(person.position, self.infection_radius)
                    for other in nearby:
                        if other.status == 'S' and np.random.random() < self.transmission_rate:
                            other.status = 'I'
                            other.infection_time = self.time
        
        # Update statistics
        counts = {'S': 0, 'I': 0, 'R': 0}
        for person in self.people:
            counts[person.status] += 1
        for status in counts:
            self.stats_history[status].append(counts[status])

# Create visualization
num_people = 200
sim = DiseaseSimulation(num_people)

# Setup plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.set_xlim(0, sim.space_size)
ax1.set_ylim(0, sim.space_size)
ax1.set_aspect('equal')

# Color mapping
color_map = {'S': 'blue', 'I': 'red', 'R': 'green'}
scatter = ax1.scatter([p.position[0] for p in sim.people],
                     [p.position[1] for p in sim.people],
                     c=[color_map[p.status] for p in sim.people])

# Statistics plot
lines = {}
for status, color in color_map.items():
    line, = ax2.plot([], [], color=color, label=status)
    lines[status] = line
ax2.set_xlim(0, 1000)
ax2.set_ylim(0, num_people)
ax2.legend()
ax2.grid(True)
ax2.set_title('Disease Progression')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Population')

def animate(frame):
    sim.update()
    
    # Update scatter plot
    scatter.set_offsets(np.c_[[p.position[0] for p in sim.people],
                             [p.position[1] for p in sim.people]])
    scatter.set_color([color_map[p.status] for p in sim.people])
    
    # Update statistics plot
    for status, line in lines.items():
        line.set_data(range(len(sim.stats_history[status])),
                     sim.stats_history[status])
    
    return scatter, *lines.values()

anim = FuncAnimation(fig, animate, frames=1000, interval=20, blit=True)
plt.show()