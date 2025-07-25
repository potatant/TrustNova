def calculate_credit_score(income, debt):
    """
    Rule-based credit score calculation.

    Parameters:
    - income (float): Annual income of the user
    - debt (float): Total outstanding debt

    Returns:
    - float: Credit score (range between 300 and 900)
    """

    if income < 0 or debt < 0:
        raise ValueError("Income and debt must be non-negative.")

    base_score = 300
    score_adjustment = (income - debt) / 1000  # Simple rule
    final_score = base_score + score_adjustment

    # Clamp the score between 300 and 900
    return max(300, min(900, final_score))
