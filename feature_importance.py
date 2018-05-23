from ClassesAndUtil.Dataset import Dataset
from simple_classifiers import get_x_y,  feature_names
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.under_sampling import RandomUnderSampler

def plot_feature_importance(X, y, names):

    rs = RandomUnderSampler(random_state=55)
    X,y = rs.fit_sample(X,y)
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
               axis=0)
    origindices = np.argsort(importances)[::-1]
    end = min(60, len(X[0]))
    indices = origindices[:end]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(end):
        print("%d. %s (%f)" % (f + 1, names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(end), importances[indices],
         color="black", yerr=std[indices], align="center")
    ordered_names = [names[ii] for ii in indices]
    plt.xticks(range(end), ordered_names[:end] ,rotation="vertical")
    # plt.xlim([-1, end])
    # plt.ylim([0,0.02])
    plt.tight_layout()
    plt.show()
    return origindices

if __name__ == '__main__':
	base_dir = os.path.join('..','YouTube-Spam-Detection','data')
	video_id_file = os.path.join(base_dir,'MyVideos','remaining_after_1000_removed.txt')

	dataset_all = Dataset(video_id_file,onlyAnnotated=True)
	known_videos = [v for v in dataset_all.all_crawled_videos if v.gold_standard!=2]
	
	X, Y = get_x_y(known_videos, feature_names)

	plot_feature_importance(X, Y, feature_names)
