from metrics import risk_bucket

def portfolio_metrics(df, pd_values, lgd):
    p = df.copy()
    p["PD"] = pd_values
    p["Risk_Bucket"] = p["PD"].apply(risk_bucket)
    p["EAD"] = p["loan_amount"]
    p["Expected_Loss"] = p["PD"] * lgd * p["EAD"]

    summary = {
        "Total_Exposure": p["EAD"].sum(),
        "Average_PD": p["PD"].mean(),
        "Total_Expected_Loss": p["Expected_Loss"].sum(),
        "High_Risk_Count": (p["Risk_Bucket"] == "High Risk").sum()
    }

    return p, summary, p["Risk_Bucket"].value_counts()
