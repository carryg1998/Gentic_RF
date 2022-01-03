from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
import pandas as pd

from Gentic_algo import Gentic_RF

boston_data = load_boston()

print(pd.DataFrame(boston_data.data, columns=boston_data.feature_names).describe())

rf = RandomForestRegressor(n_estimators=20, max_depth=8, criterion="mse", oob_score=True)
rf.fit(boston_data.data, boston_data.target)

GF = Gentic_RF(data=boston_data.data, rf_model=rf)
res = GF.gen_iter(iteration=10, rand_data=100, gen_num=5, v_range=0.1, sort=1)

print(pd.DataFrame(res, columns=boston_data.feature_names).describe())
