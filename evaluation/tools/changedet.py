import pandas as pd

# Load the CSV file with semicolons as separators
csv_filename = '/data/data0/liuyiran/Zef3d/eval_result/result_top/ZebraFish_06.txt'
df = pd.read_csv(csv_filename, header=None, sep=',')

# Reduce the values in the first column by 899
df.iloc[:, 0] -= 900
new_csv_filename = '/data/data0/liuyiran/Zef3d/eval_result/result_top/ZebraFish_06.txt'
df.to_csv(new_csv_filename, header=None, index=False)

print(f"CSV file has been updated. New file saved as {new_csv_filename}")
