with open("output_ablation.txt") as f:
    lines = f.readlines()
with open("summary.txt", "w") as out:
    out.write("".join(lines[-200:]))
