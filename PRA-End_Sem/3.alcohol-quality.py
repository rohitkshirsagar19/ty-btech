import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 🔹 1. Generate synthetic wine data
np.random.seed(42)
n = 1000
alcohol = np.random.normal(10, 1.5, n)
quality = (alcohol > 11).astype(int)  # 1 = good wine, 0 = bad wine

df = pd.DataFrame({'alcohol': alcohol, 'quality': quality})

# 🔹 2. Fit and plot Gaussian
for label in [0, 1]:
    data = df[df['quality'] == label]['alcohol']
    mu, std = norm.fit(data)
    x = np.linspace(data.min(), data.max(), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, label=f'Quality {label} (mu={mu:.2f})')

plt.title('Gaussian Distribution of Alcohol for Wine Quality')
plt.xlabel('Alcohol')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()
