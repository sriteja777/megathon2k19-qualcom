import pandas as pd
# fulltext = pd.read_csv("fulltext.csv")
abs_txt = pd.read_csv("../summaries.csv")

abstracts = [abs_txt.loc[i, "abstract"].strip().replace('\n',  ' ') for i in range(0,5)]

print(abstracts[1])
