# v2ecore/v2e_utils.py
with open("v2ecore/v2e_utils.py", "r") as f:
    c = f.read()
c = c.replace(
    "cv2.VideoWriter_fourcc(", "cv2.VideoWriter_fourcc(  # type: ignore\n        "
)
c = c.replace(
    "img = img.astype(np.float32)",
    "if img is not None:\n        img = img.astype(np.float32)",
)
with open("v2ecore/v2e_utils.py", "w") as f:
    f.write(c)

# v2ecore/synthetic/synthetic_utils.py
with open("v2ecore/synthetic/synthetic_utils.py", "r") as f:
    c = f.read()
c = c.replace("dot_sigma: float,\n):", "dot_sigma: float,\n) -> None:")
with open("v2ecore/synthetic/synthetic_utils.py", "w") as f:
    f.write(c)

# v2ecore/output/aedat2.py
with open("v2ecore/output/aedat2.py", "r") as f:
    c = f.read()
c = c.replace("self.numEventsWritten += n", "self.numEventsWritten += int(n)")
with open("v2ecore/output/aedat2.py", "w") as f:
    f.write(c)

# pyproject.toml
with open("pyproject.toml", "r") as f:
    lines = f.readlines()
with open("pyproject.toml", "w") as f:
    in_overrides = False
    for line in lines:
        if "module = [" in line and "legacy.*" in lines[lines.index(line)]:
            continue
        if "legacy.*" in line:
            continue
        if "cv2.*" in line:
            f.write('    "scipy.*",\n    "torchvision.*",\n')
        f.write(line)
