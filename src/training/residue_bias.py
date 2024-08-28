import numpy as np

# Define desired residues and their probabilities
desired_residues = {
    'E': 0.3,  # Glutamic Acid 谷氨酸
    'D': 0.3,  # Aspartic Acid 天冬氨酸
    'K': 0.2,  # Lysine 賴氨酸
    'R': 0.2   # Arginine 精氨酸
}

# Amino acid probability distribution (simplified)
amino_acid_probabilities = {
    'A': 0.05, 'C': 0.05, 'D': 0.05, 'E': 0.05, 'F': 0.05, 
    'G': 0.05, 'H': 0.05, 'I': 0.05, 'K': 0.05, 'L': 0.05,
    'M': 0.05, 'N': 0.05, 'P': 0.05, 'Q': 0.05, 'R': 0.05, 
    'S': 0.05, 'T': 0.05, 'V': 0.05, 'W': 0.05, 'Y': 0.05
}

# Function to adjust probabilities for desired residues
def adjust_probabilities(amino_acid_probabilities, desired_residues):
    for residue, bias in desired_residues.items():
        if residue in amino_acid_probabilities:
            amino_acid_probabilities[residue] += bias
    
    # Normalize probabilities to sum to 1
    total = sum(amino_acid_probabilities.values())
    for residue in amino_acid_probabilities:
        amino_acid_probabilities[residue] /= total
    
    return amino_acid_probabilities

# Adjust the probability distribution
biased_probabilities = adjust_probabilities(amino_acid_probabilities, desired_residues)

# Example sequence generation (simplified)
def generate_sequence(length, probabilities):
    amino_acids = list(probabilities.keys())
    probs = list(probabilities.values())
    sequence = ''.join(np.random.choice(amino_acids, p=probs) for _ in range(length))
    return sequence


if __name__ == "__main__":
    # Generate a sequence with the biased probabilities
    sequence_length = 100  # Example sequence length
    generated_sequence = generate_sequence(sequence_length, biased_probabilities)

    print(f"Generated Sequence: {generated_sequence}")
