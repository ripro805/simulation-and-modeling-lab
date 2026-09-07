import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

# Step 1: Generate 1000 pseudo-random numbers
np.random.seed(42)
numbers = np.random.uniform(0, 1, 1000)

# Step 2: Sort the numbers
x = np.sort(numbers)

# Number of observations
n = len(x)

# Step 3: Calculate empirical CDF
SN = np.arange(1, n + 1) / n

# Step 4: Calculate theoretical CDF
F = x

# Step 5: Calculate D+ and D-
D_plus = np.max(SN - F)
D_minus = np.max(F - np.arange(0, n) / n)

# KS statistic
D = max(D_plus, D_minus)

# Step 6: Perform KS test using scipy
ks_statistic, p_value = kstest(numbers, 'uniform')

# Significance level
alpha = 0.05

# Critical value
D_critical = 1.36 / np.sqrt(n)

# Step 7: Print results
print("Number of observations:", n)
print("D+ =", D_plus)
print("D- =", D_minus)
print("KS Test Statistic (D) =", D)
print("Critical Value =", D_critical)
print("P-value =", p_value)

# Step 8: Decision
if D < D_critical:
    print("Conclusion: Fail to reject H0.")
    print("The sequence is consistent with Uniform(0,1) distribution.")
else:
    print("Conclusion: Reject H0.")
    print("The sequence is not consistent with Uniform(0,1) distribution.")

# Step 9: Plot ECDF and theoretical CDF
plt.figure(figsize=(8, 5))

plt.step(x, SN, where='post', label='Empirical CDF S_N(x)')
plt.plot(x, F, label='Theoretical CDF F(x) = x')

# Show maximum deviation D
index = np.argmax(np.abs(SN - F))

plt.vlines(
    x[index],
    min(SN[index], F[index]),
    max(SN[index], F[index]),
    linestyle='--',
    label=f'Maximum deviation D = {D:.4f}'
)

plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.title('Kolmogorov-Smirnov Test for Uniformity')
plt.legend()
plt.grid(True)

plt.show()