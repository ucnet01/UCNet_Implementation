import pickle as pkl
import numpy as np
import os, sys

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.decomposition import PCA

from my_constants import *

def load_embeddings_and_targets(filename):
	return pkl.load(open(filename, 'rb'))

def make_numpy_arrays(embeddings_and_targets):
	all_embeddings = []
	targets = []
	for embedding, target in embeddings_and_targets:
		all_embeddings.append(np.array(embedding))
		targets.append(target)

	all_embeddings = np.array(all_embeddings)
	targets = np.array(targets)

	return all_embeddings, targets

def plot_pca(embeddings, targets, filename):
	pca = PCA(n_components=2)
	print("original shape:   ", embeddings.shape, file=sys.stderr)

	projected = pca.fit_transform(embeddings)

	print("transformed shape:", projected.shape, file=sys.stderr)

	plt.scatter(projected[:, 0], projected[:, 1], s=16,
            c=targets, edgecolor='none', alpha=0.5,
            cmap='rainbow')
	plt.xlabel('component 1')
	plt.ylabel('component 2')
	plt.colorbar();

	plt.savefig(filename, bbox_inches='tight')


def main():

	map_name_x_y = {
		tuple(LIST_ALL_AND_TWEET_FEATURES): "X_Y_{}_All_Features",
		tuple([]): "X_Y_{}_No_Features",
		tuple(LIST_UNCORRELATED): "X_Y_{}_Uncorrelated"
	}

	models_dir = os.path.join('NeuralModelsVAVDToFVC')
	annotations = 'fvc'

	feature_names = LIST_ALL_AND_TWEET_FEATURES

	pickle_name = os.path.join(models_dir, map_name_x_y[tuple(feature_names)].format(annotations))


	X, Y, known_videos = pkl.load(open(pickle_name, 'rb'))
	embeddings_and_targets = list(zip(X, Y))

	filename_plot = os.path.join('FeaturesPCA.png')

	embeddings, targets = make_numpy_arrays(embeddings_and_targets)
	plot_pca(embeddings, targets, filename_plot)

	exit(0)



	# filename_embeddings = 'Title_Embeddings_And_Target_fvc'
	# filename_plot = 'plot_pca_title.png'

	annotations = 'fvc'
	# filename_embeddings = 'Comment_Embeddings_And_Target_{}_Attention.pkl'.format(annotations)
	# filename_plot = 'plot_pca_comment_avg_attention.png'

	# filename_embeddings = 'Comment_Embeddings_And_Target_{}_Attention_Extra_Hidden_Layer.pkl'.format(annotations)
	# filename_plot = 'plot_pca_comment_avg_attention_extra_hidden.png'

	filename_embeddings = 'Comment_Embeddings_And_Target_{}_Attention_Extra_Hidden_Layer_VAVD.pkl'.format(annotations)
	filename_plot = 'plot_pca_comment_avg_attention_extra_hidden_VAVD.png'


	# filename_embeddings = 'Comment_Embeddings_And_Target_fvc'
	# filename_plot = 'plot_pca_comment_avg.png'


	embeddings_and_targets = load_embeddings_and_targets(filename_embeddings)
	embeddings, targets = make_numpy_arrays(embeddings_and_targets)
	plot_pca(embeddings, targets, filename_plot)

if __name__ == '__main__':
	main()