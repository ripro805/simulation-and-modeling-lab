import numpy as np

# --------------------------------------------------
# Part 3: Autocorrelation Test for Independence
# --------------------------------------------------

# Generate 1000 pseudo-random numbers
np.random.seed(42)
n = 1000

numbers = np.random.uniform(0, 1, n)

print("Number of generated values:", len(numbers))
print("First 10 random numbers:")
print(numbers[:10])


# --------------------------------------------------
# Function to calculate autocorrelation
# --------------------------------------------------

def autocorrelation(data, lag):
    n = len(data)

    mean = np.mean(data)

    numerator = np.sum(
        (data[:-lag] - mean) *
        (data[lag:] - mean)
    )

    denominator = np.sum(
        (data - mean) ** 2
    )

    r = numerator / denominator

    return r


# --------------------------------------------------
# Test autocorrelation for selected lags
# --------------------------------------------------

lags = [1, 2, 5, 10]

alpha = 0.05

print("\nAutocorrelation Test")
print("-" * 50)

for lag in lags:

    # Calculate autocorrelation
    r = autocorrelation(numbers, lag)

    # Standard error of autocorrelation
    sigma_r = 1 / np.sqrt(n - lag)

    # Test statistic
    z = r / sigma_r

    # Critical value for two-tailed test
    critical_value = 1.96

    print(f"\nLag = {lag}")
    print(f"Autocorrelation (r) = {r:.4f}")
    print(f"Test statistic (Z) = {z:.4f}")

    if abs(z) > critical_value:
        print("Decision: Reject H0")
        print("Conclusion: Significant dependence exists.")
    else:
        print("Decision: Do not reject H0")
        print("Conclusion: No significant dependence exists.")