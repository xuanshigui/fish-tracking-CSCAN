import pandas as pd

# Load the CSV file with semicolons as separators
csv_filename = '/data/data0/liuyiran/Zef3d/2DTo3D/convert/ZebraFish_07/GT/annotations_full.csv'
df = pd.read_csv(csv_filename, sep=';')

# Save the DataFrame to a new CSV file with commas as separators
new_csv_filename = '/data/data0/liuyiran/Zef3d/2DTo3D/convert/ZebraFish_07/GT/annotations_full2.csv'
df.to_csv(new_csv_filename, index=False)

print(f"CSV file has been updated. New file saved as {new_csv_filename}")
