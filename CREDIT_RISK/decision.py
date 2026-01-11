def credit_decision(pd):
    if pd < 0.03:
        return "Approved"
    elif pd < 0.08:
        return "Approved with Conditions"
    else:
        return "Rejected"
