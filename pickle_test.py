import pickle
import pandas as pd
from recommenders import demographic as drm

df = pd.DataFrame(
    [['Jan', 2, 10],
     ['Feb', 1, 25],
     ['Mar', 31],
     ['Apr', 22]],
    index=[0, 1, 2, 3],
    columns=['month', 'First Day', 'Last Day'])

df[:].fillna(100, inplace=True)

model = drm.DemographicRecommender(df)

outfile = open('/Users/Faisal/Development/recommender-storage/model/dr.pickle', 'wb')
pickle.dump(model, outfile)


