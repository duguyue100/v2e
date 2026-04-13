import glob

def fix(path):
    with open(path, "r") as f:
        c = f.read()
    
    numba_fallback = """
try:
    from numba import njit, jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) is False else decorator(args[0])
    jit = njit
"""
    
    c = c.replace("from numba import njit", numba_fallback)
    c = c.replace("from numba import jit", numba_fallback)
    with open(path, "w") as f:
        f.write(c)

for path in glob.glob("v2ecore/*.py"):
    fix(path)
for path in glob.glob("v2ecore/synthetic/*.py"):
    fix(path)
