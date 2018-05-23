import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

from getTweets import getTweetsAndFeatures

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
		self.init_state()

	def init_state(self):
		self.hidden_state = autograd.Variable(torch.zeros(self.num_layers * self.directions, self.batch_size, self.hidden_dim))
		self.cell_state = autograd.Variable(torch.zeros(self.num_layers * self.directions, self.batch_size, self.hidden_dim))
		self.state = (self.hidden_state, self.cell_state)

	def forward(self, sentence):
		lstm_out, self.state = self.lstm(sentence.view(len(sentence), self.batch_size, self.embedding_dim), self.state)
		lstm_out_last = lstm_out[-1]
		target_space = self.hidden_to_target(lstm_out_last)
		target_scores = F.log_softmax(target_space, dim=1)
		return target_scores


class VanillaNet(nn.Module):
	def __init__(self, input_dim, hidden_size_1, hidden_size_2, target_size, batch_size=1):
		super(VanillaNet, self).__init__()
		self.input_to_hidden1 = nn.Linear(input_dim, hidden_size_1)
		self.hidden1_to_hidden2 = nn.Linear(hidden_size_1, hidden_size_2)
		self.hidden2_to_output = nn.Linear(hidden_size_2, target_size)

	def forward(self, inputVec):
		hidden1 = F.relu(self.input_to_hidden1(inputVec))
		hidden2 = F.relu(self.hidden1_to_hidden2(hidden1))
		target_scores = F.log_softmax(self.hidden2_to_output(hidden2))
		return target_scores


# model = LSTMNet(300, 300, 2, directions=2)
model = None
# print(model)
loss_function = nn.NLLLoss()


def prepare_sentence(sentence):

	### Tokenize the sentence
	sentence = sentence.split()

	### TODO: Get wordtovec for each word in the sentence
	word_vectors = [word_to_vec(word) for word in sentence]
	tensor = torch.Tensor(word_vectors)
	return autograd.Variable(tensor)

def prepare_vector(inputVec, sz):
	ret = autograd.Variable(torch.Tensor(inputVec)).view(-1, sz)
	# print(type(ret))
	return ret


def prepare_target(target_id):
	# print(target_id)	# Should be 1 for fake, 0 for legitimate
	# target_id = np.array([target_id], dtype=np.int64)
	# target = np.zeros((target_id.size, 2), dtype=np.int64)
	# target[np.arange(target_id.size, dtype=np.int64), target_id] = 1
	# print(target)

	tensor = torch.LongTensor([target_id])#.view(-1,2)
	# print(tensor)
	return autograd.Variable(tensor)


def trainLSTM(training_data):
	optimizer = optim.SGD(model.parameters(), lr=0.01)
	for epoch in range(30):
		for sentence, target in training_data:
			model.zero_grad()
			model.hidden = model.init_state()

			sentence_in = prepare_sentence(sentence)
			target_actual = prepare_target(target)

			target_predicted_scores = model(sentence_in)

			loss = loss_function(target_predicted_scores, target_actual)
			print("loss: ", loss)
			loss.backward()
			optimizer.step()


def trainVanilla(training_data):
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	for epoch in range(100):
		for inputVec, target in training_data:
			model.zero_grad()

			inputVec = prepare_vector(inputVec, len(inputVec))
			target_actual = prepare_target(target)

			target_predicted_scores = model(inputVec)
			# print("target_predicted_scores: ", target_predicted_scores)
			# print("target_actual", target_actual)

			loss = loss_function(target_predicted_scores, target_actual)
			print("loss: ", loss)
			# exit(0)
			loss.backward()
			optimizer.step()

def testVanilla(testing_data):
	for inputVec, target in testing_data:
		inputVec = prepare_vector(inputVec, len(inputVec))
		# print(inputVec)
		y_predicted = model(inputVec)
		print(y_predicted.data.tolist())
		_, predicted = torch.max(y_predicted.data, 1)

		y_true = target

		print(y_true, predicted.tolist()[0][0])


def main():
	global model
	dataset = getTweetsAndFeatures('tweet_dataset.tsv')

	X = []
	Y = []
	for x,y in dataset:
		X.append(x)
		Y.append(y)

	# print(len(X[0]))
	model = VanillaNet(len(X[0]), int(len(X[0])**0.5), int(len(X[0])**0.25), 2)
	print(model)
	# exit(0)


	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

	trainVanilla(zip(X_train, Y_train))

	for param in model.parameters():
		print(param.data)

	exit(0)


	testVanilla(zip(X_test, Y_test))

	# exit(0)

	

if __name__ == '__main__':
	main()