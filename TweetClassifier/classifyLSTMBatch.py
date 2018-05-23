"""
Not correct as left working on this.

The problem is that in mini-batches, all the sentences in the batch should be of same length.
So, we will have to keep a maximum length of sentence and pad all sentences to that length.
Didn't do it. 

"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as weight_init

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

from getTweets import getTweetEmbeddingLists

class LSTMNet(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, target_size, num_layers=1, directions=1, batch_size=1):
		super(LSTMNet, self).__init__()
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
		self.hidden_to_target = nn.Linear(hidden_dim, target_size)

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

	def forward(self, sentences):
		lstm_out, self.state = self.lstm(sentences.view(len(sentences[0]), self.batch_size, self.embedding_dim), self.state)
		lstm_out_last = lstm_out[-1]
		# print(lstm_out_last)
		target_space = self.hidden_to_target(lstm_out_last)
		target_scores = F.log_softmax(target_space)
		return target_scores

model = None
loss_function = nn.NLLLoss()
batch_size = 16


def prepare_vector(inputVec):
	print(len(inputVec[0]), len(inputVec[0][0]))
	ret = autograd.Variable(torch.Tensor(inputVec).view(batch_size, len(inputVec[0]), len(inputVec[0][0])))
	print(ret.size())
	return ret

def prepare_target(target_ids):
	tensor = torch.LongTensor([target_ids]).view(batch_size, 1)
	return autograd.Variable(tensor)

def trainLSTM(training_data):
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	for epoch in range(10):
		for sentence, target in training_data:
			model.zero_grad()
			model.hidden = model.init_state()

			sentence_in = prepare_vector(sentence)
			target_actual = prepare_target(target)
			
			target_predicted_scores = model(sentence_in)
			
			loss = loss_function(target_predicted_scores.view(-1), target_actual.view(-1))
			print("loss: ", loss)
			loss.backward()
			optimizer.step()

def trainLSTMBatch(X, Y):
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	for epoch in range(10):
		for i in range(0,len(X), batch_size):
			model.zero_grad()
			batch_x = prepare_vector(X[i:i+batch_size])
			batch_y = prepare_target(Y[i:i+batch_size])
			outputs = model(batch_x)
			loss = loss_function(outputs, batch_y)
			loss.backward()
			optimizer.step()

def testLSTMBatch(X, Y):
	for i in range(0,len(X), batch_size):
		batch_x = X[i:i+batch_size]
		batch_y = Y[i:i+batch_size]
		outputs = model(batch_x)
		loss = loss_function(outputs, batch_y)
		loss.backward()
		optimizer.step()


def testLSTM(testing_data):
	for inputVec, target in testing_data:
		inputVec = prepare_vector(inputVec)
		y_predicted = model(inputVec)
		print(y_predicted.data.tolist())
		_, predicted = torch.max(y_predicted.data, 1)

		y_true = target

		print(y_true, predicted.tolist()[0][0])


def main():
	global model
	dataset = getTweetEmbeddingLists('tweet_dataset.tsv')

	X = []
	Y = []
	for x,y in dataset:
		X.append(x)
		Y.append(y)

	model = LSTMNet(300, 300, 2, batch_size=batch_size,directions=1)
	print(model)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

	# trainLSTM(zip(X_train, Y_train))
	trainLSTMBatch(X_train, Y_train)
	testLSTM(zip(X_test, Y_test))


if __name__ == '__main__':
	main()