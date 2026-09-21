import subprocess

print("Executing harness...")
with open("output2.txt", "w") as f:
    subprocess.run(["python3", "run_harness.py"], stdout=f, stderr=f)
print("Done")
