import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


infile = "output.jsonl"
date = "2024-03"  # used in the plot

durations = []

with open(infile) as f:
    for line in f:
        data = json.loads(line)
        l = data["left"]["finish"]
        r = data["right"]["finish"]
        v = data["timestamp"]
        durations.append(v - max(l, r))

print(
    f"Avg: {np.mean(durations)}, Median: {np.median(durations)}, Max: {np.max(durations)}"
)

# Define the new cutoff and number of bins
cutoff = 200.0  # New cutoff value
num_bins_inside_cutoff = 20  # Number of bins from 0 to cutoff

for i, n in enumerate(durations):
    if n > cutoff:
        durations[i] = cutoff + 0.5 * cutoff / num_bins_inside_cutoff

# Create bin edges from 0 to cutoff, with the specified number of bins
bin_edges = np.linspace(0, cutoff, num_bins_inside_cutoff + 1)

# Adjusting the overflow bin to end at 110
overflow_cap = (
    cutoff + cutoff / num_bins_inside_cutoff
)  # Adjust as needed based on distribution
bin_edges = np.append(bin_edges, overflow_cap)

# Create the plot with custom bins
sns.histplot(
    durations, bins=bin_edges, kde=False
)  # Turn off KDE for clearer bar visibility
plt.title(f'Distribution of "time to vote" {date}')
plt.xlabel("Duration (seconds)")
plt.ylabel("Frequency")

# Highlight the overflow bin
plt.axvline(x=cutoff, color="red", linestyle="--")
plt.text(
    cutoff + 1, plt.ylim()[1] * 0.9, "Overflow", color="red", ha="left"
)  # Adjust text alignment

# Customizing x-axis labels to hide the "110"
ax = plt.gca()  # Get current axis
labels = [item.get_text() for item in ax.get_xticklabels()]
if "110" in labels:
    labels[labels.index("110")] = ""  # Replace "110" with an empty string
ax.set_xticklabels(labels)

# Ensure nothing is cut off in the plot
plt.tight_layout()

# Save the plot to a file with high resolution
plt.savefig(f"duration_distribution_time_to_vote_{date}.png", dpi=300)
