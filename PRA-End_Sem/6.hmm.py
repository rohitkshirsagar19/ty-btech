import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer

# Example continuous observations
'''
Each sublist like [30, 40] represents an observation.

    30 → Temperature (°C)

    40 → Humidity (%)

So:

    [30, 40] means 30°C and 40% humidity.
'''
obs_cont = np.array([
    [30, 40], [32, 42], [35, 45],   # Sunny
    [25, 60], [22, 65], [23, 63],   # Cloudy
    [20, 85], [18, 90], [17, 88]    # Rainy
])

# Convert continuous to discrete bins
binner = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='uniform')
obs_disc = binner.fit_transform(obs_cont).astype(int)

# Combine binned features into a single value (1D)
obs_flat = obs_disc[:, 0] * 6 + obs_disc[:, 1]

# Initialize Discrete HMM
model_disc = hmm.MultinomialHMM(n_components=3, random_state=42, n_iter=100)
model_disc.fit(obs_flat.reshape(-1, 1))

# Predict hidden states
states_disc = model_disc.predict(obs_flat.reshape(-1, 1))

print("🔹 Discrete HMM predicted states:")
print(states_disc)


# Use same continuous observations
obs = obs_cont

# Initialize Gaussian HMM
model_cont = hmm.GaussianHMM(n_components=3, covariance_type="full", random_state=42, n_iter=100)
model_cont.fit(obs)

# Predict hidden states
states_cont = model_cont.predict(obs)

print("🔹 Continuous HMM predicted states:")
print(states_cont)

print("\n🔁 Comparison:")
for i in range(len(obs)):
    print(f"Obs {i+1}: Temp={obs[i][0]}, Humidity={obs[i][1]} | Discrete State={states_disc[i]}, Continuous State={states_cont[i]}")


