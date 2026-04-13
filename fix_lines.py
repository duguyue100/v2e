with open("v2ecore/emulator.py") as f:
    lines = f.readlines()

def del_range(start, end):
    for i in range(start-1, end):
        lines[i] = "\n"

# Delete try-except block for file outputs (292 to 337)
del_range(292, 337)

# Delete prepare_storage body (356 to 376)
del_range(356, 376)
lines[354] += "        pass\n"  # add pass to def prepare_storage

# Delete closing (389 to 400)
del_range(389, 400)

with open("v2ecore/emulator.py", "w") as f:
    f.writelines(lines)
