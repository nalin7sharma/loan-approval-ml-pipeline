import pandas as pd

df = pd.read_csv("outputs/results.csv")
print(df.sort_values(by="F1", ascending=False))