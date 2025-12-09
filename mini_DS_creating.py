import pandas as pd

# 1. Read the CSV file
df = pd.read_csv("ted_talks_en.csv")

# 2. Take the first 50 rows
mini_df = df.head(50)

# 3. Save the mini-dataset
mini_df.to_csv("mini_dataset.csv", index=False)

print("Mini dataset created: mini_dataset.csv")
