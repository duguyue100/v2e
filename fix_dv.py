with open("v2ecore/output/aedat4.py", "r") as f:
    c = f.read()

c = c.replace("import dv_processing as dv", "")

# inside __init__
c = c.replace(
    "def __init__(self, filepath: str):",
    """def __init__(self, filepath: str):
        try:
            import dv_processing as dv
            self.dv = dv
        except ImportError:
            raise Exception("dv-processing is required for AEDAT-4.0 output. Install it with: pip install v2e[aedat4]")
"""
)

# replace dv with self.dv
c = c.replace("dv.", "self.dv.")

with open("v2ecore/output/aedat4.py", "w") as f:
    f.write(c)
