import pandas as pd

csv1 = pd.read_csv("./tmp/output_1.csv")
csv2 = pd.read_csv("./tmp/output_2.csv")
csv3 = pd.read_csv("./tmp/output_3.csv")

sim = csv1["ans"] == csv3["ans"]
print(sum(sim) / len(sim))

# 1-2 : 93.1%
# 2-3 : 92.6%
# 1-3 : 94.3%
