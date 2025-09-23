# compute F1-score
def compute_f1(precision, recall):
    """Compute F1"""
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1
