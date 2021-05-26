import pandas as pd

fielders = pd.read_csv("fielding.csv", sep=";")
keepers = []
for name in fielders['Name']:
    if '†' in name:
        if name not in keepers:
            keepers.append(name)

print(keepers)
