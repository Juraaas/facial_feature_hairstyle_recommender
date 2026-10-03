import pandas as pd, numpy as np

df  = pd.read_csv("dataset/hair_dataset/labels_with_hard.csv")
rng = np.random.default_rng(99)
mask = rng.random(len(df)) < 0.15
test = df[mask]
test.to_csv("dataset/hair_dataset/fixed_test.csv", index=False)
print(f"Fixed test set: {len(test)} images")