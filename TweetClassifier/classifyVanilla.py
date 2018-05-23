import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as weight_init

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

from getTweets import getTweetsAndFeatures


class VanillaNet(nn.Module):
	def __init__(self, input_dim, hidden_size_1, hidden_size_2, target_size, batch_size=1):
		super(VanillaNet, self).__init__()
		self.input_to_hidden1 = nn.Linear(input_dim, hidden_size_1)
		self.hidden1_to_hidden2 = nn.Linear(hidden_size_1, hidden_size_2)
		self.hidden2_to_output = nn.Linear(hidden_size_2, target_size)

		for param in self.parameters():
			# print(param.data)
			try:
				weight_init.xavier_normal(param.data)
			except ValueError:
				weight_init.constant(param.data, 0)
			# print(param.data)

		# exit(0)

	def forward(self, inputVec):
		hidden1 = F.relu(self.input_to_hidden1(inputVec))
		hidden2 = F.relu(self.hidden1_to_hidden2(hidden1))
		target_scores = F.log_softmax(self.hidden2_to_output(hidden2))
		return target_scores


model = None
loss_function = nn.NLLLoss()

def prepare_vector(inputVec, sz):
	ret = autograd.Variable(torch.Tensor(inputVec)).view(-1, sz)
	return ret

def prepare_target(target_id):
	tensor = torch.LongTensor([target_id])
	return autograd.Variable(tensor)

def trainVanilla(training_data):
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	for epoch in range(100):
		for inputVec, target in training_data:
			model.zero_grad()

			inputVec = prepare_vector(inputVec, len(inputVec))
			target_actual = prepare_target(target)

			target_predicted_scores = model(inputVec)

			loss = loss_function(target_predicted_scores, target_actual)
			print("loss: ", loss)
			loss.backward()
			optimizer.step()

def testVanilla(testing_data):
	y_true_list = []
	y_predicted_list = []
	for inputVec, target in testing_data:
		inputVec = prepare_vector(inputVec, len(inputVec))
		y_predicted = model(inputVec)
		print(y_predicted.data.tolist())
		_, predicted = torch.max(y_predicted.data, 1)

		y_true = target
		y_true_list.append(y_true)

		y_predicted = predicted.tolist()[0][0]
		y_predicted_list.append(y_predicted)

		print(y_true, y_predicted)

	print(confusion_matrix(y_true_list, y_predicted_list))


def main():
	global model
	dataset = getTweetsAndFeatures('tweet_dataset.tsv')

	X = []
	Y = []
	for x,y in dataset:
		X.append(x)
		Y.append(y)

	model = VanillaNet(len(X[0]), int(len(X[0])**0.5), int(len(X[0])**0.25), 2)
	print(model)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

	trainVanilla(zip(X_train, Y_train))

	for param in model.parameters():
		print(param.data)
	
	testVanilla(zip(X_test, Y_test))

if __name__ == '__main__':
	main()