import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Stock:
    def __init__(self, price, market_cap, volatility, sector_beta):
        self.price = price
        self.market_cap = market_cap
        self.volatility = volatility
        self.sector_beta = sector_beta  # Correlation with sector
        self.velocity = 0
        self.history = [price]

class MarketSimulator:
    def __init__(self, num_stocks=100):
        # Initialize stocks with random parameters
        self.stocks = []
        sectors = ['Tech', 'Finance', 'Energy', 'Healthcare']
        
        for _ in range(num_stocks):
            price = np.random.lognormal(4, 0.5)  # Random initial price
            market_cap = price * np.random.lognormal(10, 1)  # Random market cap
            volatility = np.random.uniform(0.1, 0.4)  # Random volatility
            sector_beta = np.random.uniform(0.5, 1.5)  # Random sector correlation
            
            self.stocks.append(Stock(price, market_cap, volatility, sector_beta))
    
    def calculate_market_forces(self, stock_idx):
        """Calculate forces on a stock using n-body like interactions"""
        force = 0
        main_stock = self.stocks[stock_idx]
        
        for i, other_stock in enumerate(self.stocks):
            if i != stock_idx:
                # Distance in price-marketcap space
                price_diff = other_stock.price - main_stock.price
                cap_ratio = other_stock.market_cap / main_stock.market_cap
                
                # Gravitational-like force
                if abs(price_diff) > 1e-6:  # Avoid division by zero
                    # Larger stocks have more influence
                    force += (cap_ratio * price_diff) / (abs(price_diff) ** 2)
                    
                    # Sector correlation effects
                    if abs(other_stock.sector_beta - main_stock.sector_beta) < 0.2:
                        force *= 1.5  # Stronger influence within sector
        
        # Add market sentiment (random walk component)
        force += np.random.normal(0, main_stock.volatility)
        
        return force

    def update(self, dt=0.01):
        """Update all stock prices"""
        for i in range(len(self.stocks)):
            stock = self.stocks[i]
            
            # Calculate force
            force = self.calculate_market_forces(i)
            
            # Update velocity (price momentum)
            stock.velocity = 0.99 * stock.velocity + force * dt
            
            # Update price
            stock.price += stock.velocity
            
            # Ensure price stays positive
            stock.price = max(stock.price, 0.01)
            
            # Record history
            stock.history.append(stock.price)

# Create simulation
sim = MarketSimulator(num_stocks=50)

# Run simulation
num_steps = 1000
for _ in range(num_steps):
    sim.update()

# Visualization
plt.figure(figsize=(15, 8))

# Plot top 5 largest stocks by market cap
largest_stocks = sorted(sim.stocks, key=lambda x: x.market_cap, reverse=True)[:5]

for i, stock in enumerate(largest_stocks):
    plt.plot(stock.history, label=f'Stock {i+1} (Cap: {stock.market_cap:.0f}B)')

plt.title('Market Simulation using N-Body Dynamics')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print some market statistics
for i, stock in enumerate(largest_stocks):
    returns = np.diff(stock.history) / stock.history[:-1]
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    print(f"Stock {i+1}:")
    print(f"  Final Price: ${stock.price:.2f}")
    print(f"  Volatility: {volatility:.2%}")
    print(f"  Total Return: {(stock.price/stock.history[0] - 1):.2%}")