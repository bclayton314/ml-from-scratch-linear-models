import numpy as np
import pandas as pd


n_points = 100
slope = 3
intercept = 5

np.random.seed(42)
x = np.random.randint(0, 100, n_points)
noise = np.random.normal(0, 40, n_points)


y = slope * x + intercept + noise


y = y.astype(int)

df = pd.DataFrame({
    "x": x,
    "y": y
})

df.to_csv("data.csv", index=False)

print("data.csv created successfully!")
print(df.head())