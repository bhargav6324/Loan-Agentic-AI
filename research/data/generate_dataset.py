import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=5000):
    np.random.seed(42)
    
    # 1. Base demographics
    age = np.random.randint(21, 65, size=num_samples)
    # Gender: 0 for Male, 1 for Female
    gender = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
    
    # 2. Financials
    # Income (slightly correlated with age, but NOT gender to keep ground truth fair)
    monthly_income = np.random.normal(50000 + (age - 21) * 1000, 20000, size=num_samples)
    monthly_income = np.clip(monthly_income, 15000, 300000)
    
    loan_amount = np.random.uniform(50000, 3000000, size=num_samples)
    
    employment_type = np.random.choice(["salaried", "business", "self_employed"], size=num_samples, p=[0.7, 0.2, 0.1])
    emp_map = {"salaried": 3, "business": 2, "self_employed": 1}
    employment_score = np.array([emp_map[e] for e in employment_type])
    
    # NEW: Employment years (correlated with age — older people have longer tenures)
    employment_years = np.clip(
        np.random.normal((age - 21) * 0.4, 3, size=num_samples),
        0, 35
    ).astype(int)
    
    # NEW: Loan tenure in years (uniform 1-30)
    loan_tenure_years = np.random.randint(1, 31, size=num_samples)
    
    # NEW: Loan purpose
    loan_purpose = np.random.choice(
        ["home", "education", "personal", "business", "medical"],
        size=num_samples, p=[0.30, 0.15, 0.25, 0.20, 0.10]
    )
    
    # We simulate a historically biased credit scoring system where Females
    # and Young people are artificially penalized.
    base_credit_score = 650 + (monthly_income / 1000) + (employment_score * 30) + np.random.normal(0, 30, size=num_samples)
    
    # Apply bias: Females lose 100 points on average, young lose 80 points
    bias_penalty = (gender == 1) * 100 + (age < 30) * 80
    credit_score = np.clip(base_credit_score - bias_penalty, 300, 850)
    
    # 3. Ground truth repayment behavior (Historical Data)
    # The historical labels themselves contain the bias because historically, 
    # loan officers relied heavily on this biased credit score.
    annual_income = monthly_income * 12
    loan_to_income = loan_amount / annual_income
    
    true_default_risk = (
        (loan_to_income > 3).astype(float) * 0.3 +
        (credit_score < 650).astype(float) * 0.6 + 
        (credit_score < 550).astype(float) * 0.3
    )
    # Add random noise
    true_default_risk += np.random.normal(0, 0.15, size=num_samples)
    
    # Target label: historical decision (1 = default/rejected, 0 = paid/approved)
    target_default = (true_default_risk > 0.35).astype(int)
    
    defaults = np.where(target_default == 1, np.random.randint(1, 4, size=num_samples), 0)
    
    # Existing debt (to compute DTI)
    current_outstanding = np.random.uniform(0, annual_income * 0.5, size=num_samples)
    dti_ratio = current_outstanding / annual_income
    
    credit_utilisation = np.random.uniform(10, 90, size=num_samples)
    
    # ── Add Measurement Noise (prevents 1.0 accuracy) ──
    monthly_income += np.random.normal(0, np.mean(monthly_income) * 0.15, size=num_samples)
    monthly_income = np.clip(monthly_income, 15000, 500000)
    
    loan_amount += np.random.normal(0, np.mean(loan_amount) * 0.15, size=num_samples)
    loan_amount = np.clip(loan_amount, 50000, 5000000)
    
    credit_score += np.random.normal(0, np.mean(credit_score) * 0.10, size=num_samples)
    credit_score = np.clip(credit_score, 300, 850)
    
    # Recalculate derived features with noisy data
    annual_income = monthly_income * 12
    loan_to_income = loan_amount / annual_income
    dti_ratio = current_outstanding / annual_income
    
    # ── Add Label Noise (simulates ambiguous cases) ──
    flip_mask = np.random.random(size=num_samples) < 0.12
    target_default = np.where(flip_mask, 1 - target_default, target_default)

    # Create DataFrame
    df = pd.DataFrame({
        "id": range(1, num_samples + 1),
        "applicant_age": age,
        "applicant_gender": gender,
        "monthly_income": monthly_income,
        "loan_amount": loan_amount,
        "employment_type": employment_type,
        "employment_score": employment_score,
        "employment_years": employment_years,
        "loan_tenure_years": loan_tenure_years,
        "loan_purpose": loan_purpose,
        "credit_score": credit_score,
        "current_outstanding": current_outstanding,
        "dti_ratio": dti_ratio,
        "loan_to_income": loan_to_income,
        "defaults": defaults,
        "credit_utilisation": credit_utilisation,
        "target_default": target_default
    })
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "credit_risk_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {num_samples} rows at {output_path}")
    print(f"Dataset has {target_default.sum()} defaults ({target_default.mean():.1%})")

if __name__ == "__main__":
    generate_synthetic_data()
