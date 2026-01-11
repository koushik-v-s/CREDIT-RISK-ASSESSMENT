import numpy as np
import pandas as pd

def population_stability_index(expected, actual, bins=10):
    expected_perc = pd.qcut(expected, bins, duplicates="drop").value_counts(normalize=True)
    actual_perc = pd.qcut(actual, bins, duplicates="drop").value_counts(normalize=True)
    psi = ((expected_perc - actual_perc) *
           (expected_perc / actual_perc).apply(lambda x: np.log(x) if x > 0 else 0)).sum()
    return psi
