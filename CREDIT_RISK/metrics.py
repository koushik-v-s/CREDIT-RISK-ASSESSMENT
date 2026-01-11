from sklearn.metrics import roc_curve, roc_auc_score

def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return max(tpr - fpr)

def gini_coefficient(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1

def risk_bucket(pd):
    if pd < 0.03:
        return "Low Risk"
    elif pd < 0.08:
        return "Medium Risk"
    else:
        return "High Risk"
