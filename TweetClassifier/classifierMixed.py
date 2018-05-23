import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as weight_init

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from TweetClassifier.getTweets import getTweetsEmbeddingsAndFeatures
# from getTweets import getTweetsEmbeddingsAndFeatures


class MixNet(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, target_size, features_size, num_layers=1, directions=1, batch_size=1):
		super(MixNet, self).__init__()
		assert(directions==1 or directions==2)

		self.num_layers = num_layers
		self.embedding_dim = embedding_dim
		self.batch_size = batch_size
		self.directions = directions
		self.hidden_dim = hidden_dim

		bidirectional = False
		if directions == 2:
			bidirectional = True

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional)
		self.hidden_to_target = nn.Linear(hidden_dim + features_size, target_size)

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

	def forward(self, sentence, features_tweet):
		lstm_out, self.state = self.lstm(sentence.view(len(sentence), self.batch_size, self.embedding_dim), self.state)
		features_tweet = features_tweet.view(1, -1)
		lstm_out_last = lstm_out[-1]
		lstm_concate_features = torch.cat([lstm_out_last, features_tweet], dim=1)
		target_space = self.hidden_to_target(lstm_concate_features)
		target_scores = F.log_softmax(target_space)
		return target_scores

model = None
loss_function = nn.NLLLoss()

def prepare_vector(inputVec):
	ret = autograd.Variable(torch.Tensor(inputVec))
	return ret

def prepare_target(target_ids):
	tensor = torch.LongTensor([target_ids])
	return autograd.Variable(tensor)

def trainNet(training_data):
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
	for epoch in range(10):
		for (sentence, features), target in training_data:
			model.zero_grad()
			model.hidden = model.init_state()

			sentence_in = prepare_vector(sentence)
			features_in = prepare_vector(features)
			target_actual = prepare_target(target)
			
			target_predicted_scores = model(sentence_in, features_in)
			
			loss = loss_function(target_predicted_scores, target_actual)
			print("loss: ", loss)
			loss.backward()
			optimizer.step()

def testNet(testing_data):
	y_true_list = []
	y_predicted_list = []
	for (sentence, features), target in testing_data:
		sentence_in = prepare_vector(sentence)
		features_in = prepare_vector(features)

		y_predicted = model(sentence_in, features_in)
		print(y_predicted.data.tolist())
		_, predicted = torch.max(y_predicted.data, 1)

		y_true = target
		y_true_list.append(y_true)

		y_predicted = predicted.tolist()[0][0]
		y_predicted_list.append(y_predicted)

		print(y_true, y_predicted)
	matrix = confusion_matrix(y_true_list, y_predicted_list)
	accuracy = (100*(matrix[0][0]+matrix[1][1]))/(matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
	print("Accuracy:", accuracy)
	print()
	print(matrix)
	print(classification_report(y_true_list, y_predicted_list))
	return int(round(accuracy))



def main():
	global model
	dataset = getTweetsEmbeddingsAndFeatures('tweet_dataset.tsv')

	X = []
	Y = []
	for embed,features,y in dataset:
		X.append((embed, features))
		Y.append(y)

	model = MixNet(300, 300, 2, len(X[0][1]), directions=1)
	print(model)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

	trainNet(zip(X_train, Y_train))

	accuracy = testNet(zip(X_test, Y_test))

	# ... after training, save your model 
	torch.save(model.state_dict(), 'MixedNet_{}.pt'.format(accuracy))

	# To load your previously training model:
	# model.load_state_dict(torch.load('MixedNet.pt', map_location=lambda storage, loc: storage))


if __name__ == '__main__':
	main()