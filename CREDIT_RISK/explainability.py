import pandas as pd

def logistic_explain(model, feature_names):
    return pd.DataFrame({
        "Feature": feature_names,
        "Impact": model.coef_[0]
    }).sort_values(by="Impact", ascending=False)
