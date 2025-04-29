import numpy as np
import matplotlib.pyplot as plt
import math

# Define parameters
k = 0.01
max_hours = 300

# Generate data
hours = np.arange(0, max_hours + 1, 5)  # From 0 to 300 hours in steps of 5
days = hours / 24
factors = [math.exp(-k * hour) for hour in hours]

plt.figure(figsize=(10, 6))
plt.plot(hours, factors, 'b-', linewidth=2)
plt.grid(True)

# Set up twin axis for days
ax1 = plt.gca()
ax2 = ax1.twiny()
ax2.set_xlim(0, max_hours/24)  # Convert hours to days

# At 50% value (around 69.3 hours)
half_life = math.log(2) / 0.01  # Calculate when factor equals 0.5
ax1.plot(half_life, 0.5, 'ro')
ax1.annotate(f'({half_life:.1f}h ≈ {half_life/24:.1f}d, 0.5)', xy=(half_life, 0.5), xytext=(10, 10),
             textcoords='offset points', arrowprops=dict(arrowstyle='->'))

# Add labels and title
plt.title('Time Factor Decay (K=0.01)')
ax1.set_xlabel('Hours')
ax1.set_ylabel('Time Factor')
ax2.set_xlabel('Days')

# Set axis limits
ax1.set_xlim(0, max_hours)
ax1.set_ylim(0, 1.0)

plt.tight_layout()
plt.show()