import os
import random
import string
import pandas as pd

def generate_indian_name(seed):
    random.seed(seed)
    first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Riaan", "Krishna", "Ishaan", "Shaurya", "Atharva", "Ayaan", "Dhruv", "Kabir", "Ritvik", "Ananya", "Myra", "Sara", "Aadhya", "Kyra", "Diya", "Riya", "Navya", "Kavya", "Sia"]
    last_names = ["Sharma", "Verma", "Gupta", "Malhotra", "Mehta", "Singh", "Patel", "Shah", "Kumar", "Das", "Bose", "Reddy", "Rao", "Nair", "Pillai", "Iyer", "Chopra", "Joshi", "Deshmukh", "Yadav"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_pan(seed):
    random.seed(seed)
    letters = ''.join(random.choices(string.ascii_uppercase, k=5))
    numbers = ''.join(random.choices(string.digits, k=4))
    last_letter = random.choice(string.ascii_uppercase)
    return f"{letters}{numbers}{last_letter}"

def generate_aadhaar(seed):
    random.seed(seed)
    return ''.join(random.choices(string.digits, k=12))

def augment_dataset(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    print(f"Augmenting {filepath}...")
    df = pd.read_csv(filepath)
    
    # Check if already augmented
    if "applicant_pan" in df.columns:
        print(f"Skipping {filepath} - already augmented.")
        return
        
    # Generate stable IDs based on the index so it's consistent if run again
    names = []
    pans = []
    aadhaars = []
    
    for idx in range(len(df)):
        # Create a unique seed per row
        seed = f"{filepath}_{idx}"
        names.append(generate_indian_name(seed))
        pans.append(generate_pan(seed))
        aadhaars.append(generate_aadhaar(seed))
        
    df["applicant_name"] = names
    df["applicant_pan"] = pans
    df["applicant_aadhaar"] = aadhaars
    
    # Save a new version
    dir_name = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    new_filepath = os.path.join(dir_name, "indian_" + base_name)
    
    df.to_csv(new_filepath, index=False)
    print(f"Saved augmented dataset to {new_filepath}")
    
    # Also overwrite the original if you want it to be the default
    # df.to_csv(filepath, index=False)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    datasets = [
        os.path.join(base_dir, "data", "kaggle", "credit_risk_dataset.csv"),
        os.path.join(base_dir, "data", "kaggle", "processed_training_data.csv"),
        os.path.join(base_dir, "research", "data", "credit_risk_dataset.csv")
    ]
    
    for ds in datasets:
        augment_dataset(ds)
