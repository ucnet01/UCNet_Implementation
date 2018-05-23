import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as weight_init

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from ClassesAndUtil.Dataset import Dataset, DatasetScalable
from ClassesAndUtil.Video import Video
import ClassesAndUtil.textProcessor as textProcessor
from my_constants import *
from TweetClassifier.getTweets import features_tweet, embedding_list
from TweetClassifier import classifierMixed
from TweetClassifier.textUtils import TextFeaturesExtractor
from features import *

import sys
import os
import pickle as pkl

# models_dir = os.path.join('NeuralModelsAttention')
# models_dir = os.path.join('NeuralModelsVAVD')
# models_dir = os.path.join('NeuralModelsFVC')
# models_dir = os.path.join('NeuralModelsVAVDToFVCAgain')
# models_dir = os.path.join('NeuralModelsFVCToFVCAgain')
# models_dir = os.path.join('NeuralModelsTrainToFVCAgain')
# models_dir = os.path.join('NeuralModelsVAVDToVAVD')
# models_dir = os.path.join('NeuralModelsVAVDPlusFVCToFVC')
# models_dir = os.path.join('NeuralModelsBalancedVAVDPlusFVCToFVC')
models_dir = os.path.join('NeuralModelsBalancedVAVDToFVC')





fakeness_vectors_size= 30

default_directions = 1

class MixNet(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, target_size, features_size, fakeness_vectors_size, num_layers=1, directions=default_directions, batch_size=1):
		super(MixNet, self).__init__()
		assert(directions==1 or directions==2)

		self.num_layers = num_layers
		self.embedding_dim = embedding_dim
		self.batch_size = batch_size
		self.directions = directions
		self.hidden_dim = hidden_dim
		self.fakeness_vectors_size = fakeness_vectors_size
		hidden2_size = 4#int((directions*hidden_dim + features_size)**0.5)

		bidirectional = False
		if directions == 2:
			bidirectional = True

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional)
		self.fakeness_vector_to_weight = nn.Linear(self.fakeness_vectors_size, 1)
		self.hidden1_to_hidden2 = nn.Linear(directions*hidden_dim + features_size, hidden2_size)
		self.hidden2_to_target = nn.Linear(hidden2_size, target_size)

		for param in self.parameters():
			try:
				weight_init.xavier_normal(param.data)
			except ValueError:
				weight_init.constant(param.data, 0)

		self.init_state()

	def init_state(self):
		self.hidden_state = autograd.Variable(torch.zeros(self.num_layers * self.directions, self.batch_size, self.hidden_dim))
		self.cell_state = autograd.Variable(torch.zeros(self.num_layers * self.directions, self.batch_size, self.hidden_dim))
		self.state = (self.hidden_state, self.cell_state)

	def forward(self, comments, fakeness_vectors, features_tweet):
		# avg_of_comments = autograd.Variable(torch.zeros(self.directions*self.embedding_dim))
		avg_of_comments = autograd.Variable(torch.zeros(1, self.directions*self.embedding_dim))
		i = 0
		for comment, fakeness_vector in zip(comments, fakeness_vectors):
			i += 1
			lstm_out, self.state = self.lstm(comment.view(len(comment), self.batch_size, self.embedding_dim), self.state)
			fakeness_vector = fakeness_vector.view(1, -1)
			weight = F.sigmoid(self.fakeness_vector_to_weight(fakeness_vector))
			lstm_last = lstm_out[-1]
			weight = weight.expand_as(lstm_last)
			# print(avg_of_comments.size(), lstm_last.size(), weight.size(), file=sys.stderr)
			avg_of_comments += lstm_last*weight
			self.init_state()
		avg_of_comments = avg_of_comments/max(i,1)

		features_tweet = features_tweet.view(1, -1)
		avg_of_comments = avg_of_comments.view(1, -1)
		lstm_concate_features = torch.cat([avg_of_comments, features_tweet], dim=1)

		hidden2 = F.relu(self.hidden1_to_hidden2(lstm_concate_features))
		target_space = self.hidden2_to_target(hidden2)
		target_scores = F.log_softmax(target_space)

		return target_scores

	def get_comment_embeddings(self, comments, fakeness_vectors):
		avg_of_comments = autograd.Variable(torch.zeros(self.directions*self.embedding_dim))
		i = 0
		for comment, fakeness_vector in zip(comments, fakeness_vectors):
			i += 1
			lstm_out, self.state = self.lstm(comment.view(len(comment), self.batch_size, self.embedding_dim), self.state)
			fakeness_vector = fakeness_vector.view(1, -1)
			weight = F.sigmoid(self.fakeness_vector_to_weight(fakeness_vector))
			lstm_last = lstm_out[-1]
			weight = weight.expand_as(lstm_last)
			avg_of_comments += lstm_last*weight
			self.init_state()
		avg_of_comments = avg_of_comments/max(i,1)
		return avg_of_comments.data.tolist()

model = None
loss_function = nn.NLLLoss()

def prepare_vector(inputVec):
	ret = autograd.Variable(torch.Tensor(inputVec))
	return ret

def prepare_target(target_ids):
	tensor = torch.LongTensor([target_ids])
	return autograd.Variable(tensor)

def trainNet(training_data, testing_data_annotations, testing_data_fvc):
	# optimizer = optim.SGD([
	#                 {'params': model.hidden1_to_hidden2.parameters(), 'lr': 1e-2},
	#                 {'params': model.hidden2_to_target.parameters(), 'lr': 1e-2}
	#             ], lr=5*1e-4, weight_decay = 1e-5, momentum=0.9)


	# optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	for epoch in range(100):
		print("************** Epoch {} **************".format(epoch), file=sys.stderr)
		avg_loss = torch.Tensor([0])
		num_training = 0
		for ((comments, fakeness_vectors), features), target in training_data:
			num_training += 1
			model.zero_grad()
			model.init_state()

			comments_in = []
			fakeness_vectors_in = []
			for comment, fakeness_vector in zip(comments, fakeness_vectors):
				comments_in.append(prepare_vector(comment))
				fakeness_vectors_in.append(prepare_vector(fakeness_vector))
			
			features_in = prepare_vector(features)
			target_actual = prepare_target(target)
			
			target_predicted_scores = model(comments_in, fakeness_vectors_in, features_in)
			
			loss = loss_function(target_predicted_scores, target_actual)
			avg_loss += loss.data
			loss.backward()
			optimizer.step()
		loss_training = avg_loss/num_training
		loss_testing_annotations = testNet(testing_data_annotations, for_validation=True)
		loss_testing_fvc = testNet(testing_data_fvc, for_validation=True)

		print("Epoch {} : Average training loss: ".format(epoch), loss_training, file=sys.stderr)
		print("Epoch {} : Average testing on annotation loss: ".format(epoch), loss_testing_annotations, file=sys.stderr)
		print("Epoch {} : Average testing on fvc loss: ".format(epoch), loss_testing_fvc, file=sys.stderr)

		print("Epoch {} : Average training loss: ".format(epoch), loss_training)
		print("Epoch {} : Average testing on annotation loss: ".format(epoch), loss_testing_annotations)
		print("Epoch {} : Average testing on fvc loss: ".format(epoch), loss_testing_fvc)

		testNet(testing_data_annotations)

		torch.save(model.state_dict(), os.path.join(models_dir, 'CommentAvgNet_epoch_{}.pt'.format(epoch)))


def testNet(testing_data, for_validation=False):
	y_true_list = []
	y_predicted_list = []
	avg_loss = torch.Tensor([0])
	num = 0
	for ((comments, fakeness_vectors), features), target in testing_data:
		num += 1
		
		comments_in = []
		fakeness_vectors_in = []
		for comment, fakeness_vector in zip(comments, fakeness_vectors):
			comments_in.append(prepare_vector(comment))
			fakeness_vectors_in.append(prepare_vector(fakeness_vector))
		
		features_in = prepare_vector(features)

		y_predicted = model(comments_in, fakeness_vectors_in, features_in)

		if for_validation:
			target_actual = prepare_target(target)
			loss = loss_function(y_predicted, target_actual)
			avg_loss += loss.data

		else:
			_, predicted = torch.max(y_predicted.data, 1)

			y_true = target
			y_true_list.append(y_true)

			# print(predicted)

			y_predicted = predicted.tolist()#[0][0]
			y_predicted_list.append(y_predicted)

	if for_validation:
		avg_loss = avg_loss/num
		return avg_loss


	matrix = confusion_matrix(y_true_list, y_predicted_list)
	accuracy = (100*(matrix[0][0]+matrix[1][1]))/(matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
	print("Accuracy:", accuracy)
	print()
	print(matrix)
	# report = classification_report(y_true_list, y_predicted_list)
	# print(report)
	
	print("Accuracy:", accuracy, file=sys.stderr)
	print("", file=sys.stderr)
	print(matrix, file=sys.stderr)
	# print(report, file=sys.stderr)

	return int(round(accuracy))


base_dir = os.path.join('final_data')
# feature_names = LIST_UNCORRELATED
feature_names = LIST_ALL_AND_TWEET_FEATURES

def load_x_y(video_id_file, annotations):
	# models_dir = 'NeuralModelsExtraFeatures'
	pickle_name = os.path.join(models_dir, 'X_Y_{}_All_Features'.format(annotations))
	try:
		# raise FileNotFoundError
		X, Y, known_videos = pkl.load(open(pickle_name.format(annotations), 'rb'))
	except FileNotFoundError:
		dataset_all = Dataset(video_id_file,onlyAnnotated=True, annotations=annotations)
		known_videos = [v for v in dataset_all.all_crawled_videos if v.gold_standard!=2]
		X, Y = get_x_y(known_videos, feature_names)
		pkl.dump((X, Y, known_videos), open(pickle_name, 'wb'))
	return X, Y, known_videos



def load_annotated_dataset(video_id_file, annotations):
	pickle_name = os.path.join(models_dir, 'X_Y_{}_Comments_Avg_All_Features.pkl').format(annotations)
	try:
		# raise FileNotFoundError
		X, Y = pkl.load(open(pickle_name, 'rb'))
	except FileNotFoundError:
		X, Y, known_videos = load_x_y(video_id_file, annotations)

		embeddings_all_videos = []
		fakeness_vectors_all_videos = []
		for video in known_videos:
			embeddings = []
			fakeness_vectors = []
			for comment in video.get_iterator_comments(max_comments=100):
				tfe = TextFeaturesExtractor(comment.getText())
				if tfe.getFakeness():
					embeddings.append(tfe.getEmbeddingList())
					fakeness_vectors.append(tfe.getFakenessVector())
			embeddings_all_videos.append(embeddings)
			fakeness_vectors_all_videos.append(fakeness_vectors)

		num_features = len(X[0])

		X = list(zip(list(zip(embeddings_all_videos, fakeness_vectors_all_videos)), X))

		pkl.dump((X, Y), open(pickle_name, 'wb'))

	return (X, Y)

def train_on_annotations(video_id_file_annot, video_id_file_fvc, train_annotations, test_annotations):
	global model
	X, Y = load_annotated_dataset(video_id_file_annot, train_annotations)
	X_fvc, Y_fvc = load_annotated_dataset(video_id_file_fvc, test_annotations)

	X_train = X
	Y_train = Y

	X_test = X_fvc
	Y_test = Y_fvc




	# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
	
	print("Train on:", len(X_train))
	print("Test on:", len(X_test))

	print("Total number of fake videos in train set: {}".format(sum(Y_train)))
	print("Total number of fake videos in test set: {}".format(sum(Y_test)))

	model = MixNet(300, 300, 2, len(X[0][1]), fakeness_vectors_size)

	training_data = list(zip(X_train, Y_train))
	testing_data_annotations = list(zip(X_test, Y_test))
	testing_data_fvc = list(zip(X_fvc, Y_fvc))

	trainNet(training_data, testing_data_annotations, testing_data_fvc)
	testNet(testing_data_annotations)

def test_on_fvc(video_id_file):
	X, Y = load_annotated_dataset(video_id_file, 'fvc')
	return testNet(list(zip(X, Y)))


def get_comment_embeddings(video_id_file, annotations, epoch):
	X, Y = load_annotated_dataset(video_id_file, annotations)
	model_path = os.path.join(models_dir, 'CommentAvgNet_epoch_{}.pt'.format(epoch))#os.path.join(models_dir, 'CommentAvgNet_epoch_8.pt')
	model = MixNet(300, 300, 2, len(X[0][1]), fakeness_vectors_size)
	model.load_state_dict(torch.load(model_path))
	

	all_embeddings_and_target = []
	num = 0
	for ((comments, fakeness_vectors), features), target in zip(X, Y):
		num += 1
		print(num, file=sys.stderr)

		comments_in = []
		fakeness_vectors_in = []
		for comment, fakeness_vector in zip(comments, fakeness_vectors):
			comments_in.append(prepare_vector(comment))
			fakeness_vectors_in.append(prepare_vector(fakeness_vector))

		embedding_for_comments = model.get_comment_embeddings(comments_in, fakeness_vectors_in)
		all_embeddings_and_target.append((embedding_for_comments, target))

	filename = os.path.join(models_dir, 'Comment_Embeddings_And_Target_{}_Attention_Extra_Hidden_Layer_VAVD.pkl'.format(annotations))

	pkl.dump(all_embeddings_and_target, open(filename, 'wb'))
	return all_embeddings_and_target



if __name__ == '__main__':
	epoch = 2
	# get_comment_embeddings('fvc_videos.txt', 'fvc', epoch)

	# exit(0)

	annot_file = os.path.join(base_dir,'all_videos.txt')
	fvc_file = os.path.join(base_dir,'fvc_videos.txt')

	# onlyTest = True
	onlyTest = False

	if onlyTest:
		video_id_file = fvc_file
		annotations = 'fvc'
		X, Y = load_annotated_dataset(video_id_file, annotations)
		model_path = os.path.join(models_dir, 'CommentAvgNet_epoch_{}.pt'.format(epoch))
		model = MixNet(300, 300, 2, len(X[0][1]), fakeness_vectors_size)
		model.load_state_dict(torch.load(model_path))

		accuracy = testNet(list(zip(X, Y)))
		exit(0)

	# annot_file = fvc_file


	# train_on_annotations(annot_file, fvc_file, 'fvc_train', 'fvc_test')
	# train_on_annotations(annot_file, fvc_file, 'VAVD', 'fvc')
	# train_on_annotations(annot_file, fvc_file, 'train', 'fvc')
	# train_on_annotations(annot_file, annot_file, 'balanced_sud_plus_fvc_train', 'fvc_test')
	train_on_annotations(annot_file, annot_file, 'VAVD_balanced', 'fvc_test')




	# accuracy = test_on_fvc(fvc_file)
	# torch.save(model.state_dict(), 'CommentAvgNet_{}.pt'.format(accuracy))
