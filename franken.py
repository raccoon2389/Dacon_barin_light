import pandas as pd
import numpy as np

h = pd.read_csv('./Dacon_light.csv',index_col=0)
r = pd.read_csv('./Dacon_model_sel.csv',index_col=0)

h = h["hhb"]
r.update(h)
print(r)
r.to_csv('./new.csv')