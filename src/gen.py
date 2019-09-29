import pandas as pd
import numpy as np

text = pd.read_csv("../../datasets/papers.csv")

abstracts = pd.DataFrame(text['abstract'].values, columns = ['abstract'])
full_texts = pd.DataFrame(text['paper_text'].values, columns = ['paper_text'])

abstracts.to_csv('summaries.csv', index=False)
full_texts.to_csv('fulltext.csv', index=False)