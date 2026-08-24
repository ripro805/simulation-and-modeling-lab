import numpy as np
from scipy.stats import chi2

# ---------------------------------------
# Chi-Square Test for Uniformity
# ---------------------------------------

# Generate 1000 pseudo-random numbers
np.random.seed(42)
numbers = np.random.random(1000)

# Number of intervals
n = 10

# Total number of observations
N = len(numbers)

# Divide [0,1) into 10 equal intervals
bins = np.linspace(0, 1, n + 1)

# Calculate observed frequencies
observed, _ = np.histogram(numbers, bins=bins)

# Calculate expected frequency
expected = N / n

# Calculate chi-square statistic
chi_square = np.sum((observed - expected) ** 2 / expected)

# Degrees of freedom
df = n - 1

# Significance level
alpha = 0.05

# Critical value
critical_value = chi2.ppf(1 - alpha, df)

# ---------------------------------------
# Display Results
# ---------------------------------------

print("Chi-Square Test for Uniformity")
print("--------------------------------")

print("Total numbers:", N)
print("Number of intervals:", n)
print("Expected frequency:", expected)

print("\nObserved Frequencies:")
for i in range(n):
    print(
        f"[{bins[i]:.1f}, {bins[i+1]:.1f}) : "
        f"{observed[i]}"
    )

print("\nChi-square statistic:", round(chi_square, 4))
print("Degrees of freedom:", df)
print("Significance level:", alpha)
print("Critical value:", round(critical_value, 4))

# ---------------------------------------
# Decision
# ---------------------------------------

if chi_square < critical_value:
    print("\nDecision: Fail to reject H0")
    print("Conclusion: The sequence is consistent with")
    print("a uniform distribution.")
else:
    print("\nDecision: Reject H0")
    print("Conclusion: The sequence is not consistent with")
    print("a uniform distribution.")