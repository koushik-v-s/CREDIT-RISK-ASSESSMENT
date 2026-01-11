def approximate_rwa(pd, ead):
    risk_weight = min(1.0, pd * 12)  # intuition-based
    return ead * risk_weight
