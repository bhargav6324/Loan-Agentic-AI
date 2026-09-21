import subprocess

print("Executing ablation...")
with open("output_ablation.txt", "w") as f:
    subprocess.run(["python3", "research_v2/run_agentic_ablation.py", "--seeds", "42", "--configurations", "A0,A1,A1R,A2,A3"], stdout=f, stderr=f)
print("Done")
