import numpy as np
import matplotlib.pyplot as plt
import sys

def triangular_safe(x, a, b, c):
    """
    Calculates the membership degree for a triangular or ramp-shaped fuzzy set.
    Uses np.interp for robust piecewise linear interpolation, handling all cases.
    The function is 0 before 'a', rises to 1 at 'b', and falls to 0 at 'c'.
    """
    x = np.asarray(x) # Ensure x is an array for vectorized operations
    return np.interp(x, [a, b, c], [0.0, 1.0, 0.0])

def trapezoidal_safe(x, a, b, c, d):
    """
    Calculates the membership degree for a trapezoidal fuzzy set.
    The function is 0 before 'a', rises to 1 at 'b', stays 1 until 'c',
    and falls to 0 at 'd'.
    """
    x = np.asarray(x)
    return np.interp(x, [a, b, c, d], [0.0, 1.0, 1.0, 0.0])

# Define the range for input and output variables
x_temp_diff = np.linspace(0, 30, 500)
x_temp_rate = np.linspace(0, 0.3, 500)
# Expanded range and increased points for accuracy in centroid calculation, consistent with PDF figures
# Adjusted x_freq to exactly match the image's x-axis range (0 to 115)
x_freq = np.linspace(0, 115, 1000) 

mu_freq_verylow = triangular_safe(x_freq, 0, 10, 25) # From 0, peaks at 10, goes to 0 at 25
mu_freq_low = triangular_safe(x_freq, 22, 37, 54)
mu_freq_med = triangular_safe(x_freq, 45, 62, 79)
mu_freq_high = triangular_safe(x_freq, 70, 87, 104)
mu_freq_veryhigh = trapezoidal_safe(x_freq, 97, 115, x_freq.max(), x_freq.max()) 

delta_t = 10
vt = 0.2

temp_terms = {
    "малая": triangular_safe(np.array([delta_t]), 0, 0, 14.2857)[0],
    "средняя": triangular_safe(np.array([delta_t]), 10, 15, 20)[0],
    "большая": triangular_safe(np.array([delta_t]), 15, 30, 30)[0]
}

vt_terms = {
    "малая": triangular_safe(np.array([vt]), 0, 0, 0.15)[0],
    "средняя": triangular_safe(np.array([vt]), 0.05, 0.15, 0.25)[0],
    "большая": triangular_safe(np.array([vt]), 0.15, 0.3, 0.3)[0]
}

print(f"Membership degrees for Delta_t={delta_t}K:")
for term, value in temp_terms.items():
    print(f"  {term}: {value:.4f}")

print(f"\nMembership degrees for Vt={vt}K/min:")
for term, value in vt_terms.items():
    print(f"  {term}: {value:.4f}")

rules = {
    ("малая", "малая"): "очень малая",
    ("средняя", "малая"): "малая",
    ("большая", "малая"): "средняя",
    ("малая", "средняя"): "малая",
    ("средняя", "средняя"): "средняя",
    ("большая", "средняя"): "большая",
    ("малая", "большая"): "средняя",
    ("средняя", "большая"): "большая",
    ("большая", "большая"): "очень большая"
}

output_membership = {
    "очень малая": mu_freq_verylow,
    "малая": mu_freq_low,
    "средняя": mu_freq_med,
    "большая": mu_freq_high,
    "очень большая": mu_freq_veryhigh
}

# Aggregation of output fuzzy sets (Mamdani's min-max composition)
aggregated_output = np.zeros_like(x_freq)

print("\nRule Activations and Clipped Outputs:")
for (temp_term, vt_term), output_term in rules.items():
    alpha = min(temp_terms[temp_term], vt_terms[vt_term])
    clipped_output = np.minimum(output_membership[output_term], alpha)
    aggregated_output = np.maximum(aggregated_output, clipped_output)
    print(f"  Rule (Δt:{temp_term}, Vt:{vt_term}) -> F:{output_term}: Activation (alpha) = {alpha:.4f}")

# Defuzzification using the Centroid method (Center of Gravity)
defuzz_numerator = np.sum(x_freq * aggregated_output)
defuzz_denominator = np.sum(aggregated_output)
defuzzified_value = defuzz_numerator / defuzz_denominator if defuzz_denominator != 0 else 0


print(f"\nДефаззифицированное значение (центр тяжести): {defuzzified_value:.2f} Гц")

# --- Visualizations of individual membership functions (from original code) ---
# These plots will remain as they were, matching the PDF's figures 10.8, 10.9, 10.10/10.11

# Plotting input membership functions
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Разность температур (Δt) - using parameters from Figure 10.8
x_temp_diff_plot = np.linspace(0, 30, 500) 
axs[0].plot(x_temp_diff_plot, triangular_safe(x_temp_diff_plot, 0, 0, 15), label='Малая')
axs[0].plot(x_temp_diff_plot, triangular_safe(x_temp_diff_plot, 0, 15, 30), label='Средняя') # Corrected to match Fig 10.8
axs[0].plot(x_temp_diff_plot, triangular_safe(x_temp_diff_plot, 15, 30, 30), label='Большая')
axs[0].axvline(delta_t, color='gray', linestyle=':', label=f'Δt = {delta_t}K')
axs[0].set_title('Функции принадлежности: Разность температур (Δt)')
axs[0].set_xlabel('Δt (K)')
axs[0].set_ylabel('Степень принадлежности')
axs[0].legend()
axs[0].grid(True)

# Скорость изменения температуры (Vt) - using parameters from Figure 10.9
x_temp_rate_plot = np.linspace(0, 0.3, 500) 
axs[1].plot(x_temp_rate_plot, triangular_safe(x_temp_rate_plot, 0, 0, 0.15), label='Малая')
axs[1].plot(x_temp_rate_plot, triangular_safe(x_temp_rate_plot, 0, 0.15, 0.3), label='Средняя') # Corrected to match Fig 10.9
axs[1].plot(x_temp_rate_plot, triangular_safe(x_temp_rate_plot, 0.15, 0.3, 0.3), label='Большая')
axs[1].axvline(vt, color='gray', linestyle=':', label=f'Vt = {vt}K/мин')
axs[1].set_title('Функции принадлежности: Скорость изменения температуры (Vt)')
axs[1].set_xlabel('Vt (K/мин)')
axs[1].set_ylabel('Степень принадлежности')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# Plotting output membership functions - using parameters from Figure 10.10/10.11
plt.figure(figsize=(10, 6))
plt.plot(x_freq, mu_freq_verylow, label='Очень малая')
plt.plot(x_freq, mu_freq_low, label='Малая')
plt.plot(x_freq, mu_freq_med, label='Средняя')
plt.plot(x_freq, mu_freq_high, label='Большая')
plt.plot(x_freq, mu_freq_veryhigh, label='Очень большая') # Now uses trapezoidal_safe

plt.title('Функции принадлежности: Частота вращения компрессора (F)')
plt.xlabel('F (Гц)')
plt.ylabel('Степень принадлежности')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plotting the aggregated output and centroid to match the provided image
plt.figure(figsize=(10, 3)) # Adjusted figure size to be more compact like the image
ax = plt.gca() # Get current axes
ax.set_facecolor('#E0E0E0') # Light grey background
plt.plot(x_freq, aggregated_output, color='blue', linewidth=0) # No line, just fill
plt.fill_between(x_freq, 0, aggregated_output, color='blue') # Solid blue fill
plt.axvline(float(defuzzified_value), color='red', linestyle='-', linewidth=4) # Thicker red line
plt.title('') # Remove title
plt.xlabel('') # Remove x-label
plt.ylabel('') # Remove y-label
plt.legend().set_visible(False) # Remove legend
plt.grid(False) # Remove grid
plt.xticks([0, 115]) # Only show 0 and 115 on x-axis
plt.yticks([]) # Remove y-axis ticks
plt.xlim(0, 115) # Set x-axis limits explicitly
plt.ylim(0, 1.0) # Set y-axis limits explicitly
plt.tight_layout()
plt.show()