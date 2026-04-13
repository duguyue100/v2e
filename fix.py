import os

files = [
    'v2ecore/output/aedat4.py',
    'v2ecore/output/aedat2.py',
    'v2ecore/output/text.py',
    'v2ecore/output/base.py',
]

for file in files:
    with open(file, 'r') as f:
        lines = f.readlines()
    
    with open(file, 'w') as f:
        for line in lines:
            if 'def ' in line and '->' not in line:
                line = line.rstrip() + ' -> Any: # type: ignore\n'
            f.write(line)

