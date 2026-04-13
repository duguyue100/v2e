import re

files = [
    'v2ecore/output/aedat4.py',
    'v2ecore/output/aedat2.py',
    'v2ecore/output/text.py',
    'v2ecore/output/base.py',
    'v2ecore/output/hdf5.py'
]

for file in files:
    with open(file, 'r') as f:
        content = f.read()

    # Remove the bad replacement I did if any:
    content = re.sub(r' -> Any: # type: ignore\n', ':\n', content)
    
    with open(file, 'w') as f:
        f.write(content)
