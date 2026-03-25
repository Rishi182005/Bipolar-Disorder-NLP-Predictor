import pandas as pd

# load existing final_project_data
df = pd.read_csv("project_files/final_project_data.csv")

mood_map = {"depressive": 0, "stable": 1, "manic": 2}
df["mood_encoded"] = df["mood"].map(mood_map)
df["predicted_mood_encoded"] = df["predicted_mood"].map(mood_map)

df.to_csv("project_files/final_project_data.csv", index=False)
print("Saved final_project_data.csv with encoded columns.")
