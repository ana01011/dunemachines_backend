# Fix cortisol.py to handle None urgency
import fileinput

for line in fileinput.input('app/neurochemistry/hormones/cortisol.py', inplace=True):
    if "response = self.sensitivity * urgency * 0.5" in line:
        print("            response = self.sensitivity * (urgency or 0.5) * 0.5")
    elif "stress_from_urgency = urgency * 0.3" in line:
        print("            stress_from_urgency = (urgency or 0.0) * 0.3")
    else:
        print(line, end='')
