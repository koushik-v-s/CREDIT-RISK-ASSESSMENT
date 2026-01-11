def apply_stress(df, level):
    stressed = df.copy()

    if level == "Mild":
        stressed["income"] *= 0.95
        stressed["credit_score"] -= 20
    elif level == "Severe":
        stressed["income"] *= 0.85
        stressed["credit_score"] -= 50

    return stressed
