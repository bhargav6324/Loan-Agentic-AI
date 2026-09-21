import subprocess

print("Executing analysis...")
with open("analyze_output.txt", "w") as f:
    subprocess.run(["python3", "analyze_differences.py"], stdout=f, stderr=f)
print("Done")
