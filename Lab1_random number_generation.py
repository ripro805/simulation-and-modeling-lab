import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# Linear Congruential Generator Parameters
# -----------------------------------------

a = 1664525
c = 1013904223
m = 2**32
X0 = 12345

# Number of random numbers
n = 10000

# -----------------------------------------
# Generate pseudo-random numbers
# -----------------------------------------

X = X0
random_numbers = []

for i in range(n):
    X = (a * X + c) % m
    R = X / m
    random_numbers.append(R)

# Convert to NumPy array
random_numbers = np.array(random_numbers)

# -----------------------------------------
# Calculate sample mean and variance
# -----------------------------------------

sample_mean = np.mean(random_numbers)

# ddof=1 gives sample variance
sample_variance = np.var(random_numbers, ddof=1)

# -----------------------------------------
# Theoretical values for U(0,1)
# -----------------------------------------

theoretical_mean = 0.5
theoretical_variance = 1 / 12

# -----------------------------------------
# Display results
# -----------------------------------------

print("Number of random numbers:", n)

print("\nFirst 10 random numbers:")
print(random_numbers[:10])

print("\nSample Mean:")
print(sample_mean)

print("\nTheoretical Mean:")
print(theoretical_mean)

print("\nSample Variance:")
print(sample_variance)

print("\nTheoretical Variance:")
print(theoretical_variance)

# -----------------------------------------
# Compare results
# -----------------------------------------

mean_error = abs(sample_mean - theoretical_mean)
variance_error = abs(sample_variance - theoretical_variance)

print("\nMean Absolute Error:")
print(mean_error)

print("\nVariance Absolute Error:")
print(variance_error)

# -----------------------------------------
# Plot Histogram
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    random_numbers,
    bins=10,
    edgecolor="black"
)

plt.title("Histogram of 10,000 Pseudo-Random Numbers")
plt.xlabel("Random Number")
plt.ylabel("Frequency")

plt.show()