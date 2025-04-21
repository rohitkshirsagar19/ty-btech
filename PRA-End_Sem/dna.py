import numpy as np
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mode
import matplotlib.pyplot as plt

# Example DNA sequences and their labels
seqs = ["ATGCG", "CGTGA", "GCGTT", "ATGCC", "TTGCA"]
labels = ["EEIII", "EIIII", "EEEII", "EEIII", "IIEEE"]

# Mapping of DNA bases to numbers
obs_map = {"A": 0, "T": 1, "G": 2, "C": 3}
# Mapping of state labels to numbers (Exon = 0, Intron = 1)
state_map = {"E": 0, "I": 1}

# Convert DNA sequences to numbers
X = []
for seq in seqs:
    for c in seq:
        X.append(obs_map[c])
X = np.array(X).reshape(-1, 1)  # Reshape to a column vector

# Convert the label sequences to numbers (0 for Exon, 1 for Intron)
y = []
for lab in labels:
    for c in lab:
        y.append(state_map[c])
y = np.array(y)

# Calculate sequence lengths (the number of bases in each sequence)
lengths = [len(seq) for seq in seqs]

# Initialize and train the HMM model
model = hmm.MultinomialHMM(n_components=2, n_iter=100).fit(X, lengths)

# Decode the hidden states for the input sequence
_, hidden = model.decode(X)

# Simplified version of mapping the hidden states to the most frequent true labels
mapped = {}
for s in np.unique(hidden):
    mask = (hidden == s)
    true_labels_for_s = y[mask]
    
    # Get the most frequent true label
    most_common_label_result = mode(true_labels_for_s)
    
    # If the result is a scalar (i.e., only one mode), use it directly
    if isinstance(most_common_label_result.mode, np.ndarray):
        most_common_label = most_common_label_result.mode[0]  # If multiple modes, use the first one
    else:
        most_common_label = most_common_label_result.mode  # Single mode value
    
    mapped[s] = most_common_label

# Create the predicted labels based on the mapped dictionary
pred = np.array([mapped[s] for s in hidden])

# Print the accuracy and classification report
print("Accuracy:", accuracy_score(y, pred))
print(classification_report(y, pred, target_names=["Exon", "Intron"]))

# Plot the true and predicted states
plt.plot(y, label="True", marker='o')
plt.plot(pred, label="Pred", linestyle='--', marker='x')
plt.legend()
plt.title("DNA State Prediction")
plt.xlabel("Position")
plt.ylabel("State")
plt.show()
