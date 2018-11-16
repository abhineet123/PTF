__author__ = 'Tommy'
from Histogram import *
from NoFeature import *

def getFeatureObject(feature, multi_approach):
    if feature == 'none':
        feature_obj = NoFeature(multi_approach=multi_approach)
    elif feature == 'hoc':
        feature_obj = Histogram(multi_approach=multi_approach)
    else:
        raise SystemExit('Invalid feature type ', feature)

    return feature_obj
