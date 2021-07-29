import pickle
from utilities import constants as const


def execute(model=None):
    print(model.loc[0:1])
    outfile = open(const.PICKLE_DEMOGRAPHIC_MODEL, 'wb')
    pickle.dump(model, outfile)
