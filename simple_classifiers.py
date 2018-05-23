from sys import argv
import os, sys
import json
import numpy as np
import pickle as pkl
import csv

from ClassesAndUtil.Dataset import Dataset, DatasetScalable
from ClassesAndUtil.Video import Video
import ClassesAndUtil.textProcessor as textProcessor
from my_constants import *
from TweetClassifier.getTweets import features_tweet, embedding_list
from TweetClassifier import classifierMixed

import sklearn
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import graphviz

import torch

# base_dir = os.path.join('..','YouTube-Spam-Detection','data')
# base_dir = os.path.join('..','YouTube-Spam-Detection','Entdata')
base_dir = os.path.join('final_data')

MAX_COMMENTS = 100
class_weight={0:0.228, 1:0.772}
# class_weight={0:0.1, 1:0.9}

tweet_classfier_mixnet = classifierMixed.MixNet(300, 300, 2, 308, directions=1)
tweet_classfier_mixnet.load_state_dict(torch.load(TWEET_MODEL_PATH))

# feature_names = LIST_ALL_AND_TWEET_FEATURES
feature_names = LIST_UNCORRELATED


clfs = [("SVM_RBF", svm.SVC(kernel="rbf", probability=True, class_weight=class_weight)),
			("SVM_Lin", svm.SVC(kernel="linear", probability=True, class_weight=class_weight)),
			("RF", ensemble.RandomForestClassifier(n_estimators=20, class_weight=class_weight)),
			# ("AB", ensemble.AdaBoostClassifier(n_estimators=10, class_weight=class_weight)),
			# ("GradBoost", ensemble.GradientBoostingClassifier(class_weight=class_weight)),
			("Logistic", sklearn.linear_model.LogisticRegression(class_weight=class_weight)),
			("DecisionTreeClassifier", tree.DecisionTreeClassifier(criterion="entropy",min_samples_leaf = 0.05, min_samples_split = 0.2, class_weight=class_weight))
		]


def get_x(video, feature_names):
	features = []
	for feature in feature_names:
		if RATIO_VIOLENT_WORDS == feature:
			features.append(textProcessor.ratio_violent_words(video.get_title()))
		elif RATIO_CAPS == feature:
			features.append(textProcessor.ratio_caps(video.get_title()))
		elif HAS_CLICKBAIT_PHRASE == feature:
			features.append(int(textProcessor.has_clickbait_phrase(video.get_title())))
		
		elif HAS_COMMENTS == feature:
			features.append(int(video.get_num_comments() == 0))
		elif COMMENTS_FAKENESS == feature:
			fakeness = video.get_comments_fakeness(max_comments=MAX_COMMENTS)
			features.append(fakeness)

		elif DISLIKE_LIKE_RATIO == feature:
			features.append((1.0*video.get_dislike_count())/max(1, video.get_like_count()))


		elif FIRST_WORD_CAPS == feature:
			features.append(int(textProcessor.first_word_capital(video.get_title())))
		elif COMMENTS_INAPPROPRIATENESS == feature:
			features.append(video.get_comments_inappropriateness(max_comments=MAX_COMMENTS))
		elif COMMENTS_CONVERSATION_RATIO == feature:
			features.append(video.get_conversation_ratio(recursive=False,max_comments=MAX_COMMENTS))


		elif TWEET_CLASSIFIER_TITLE == feature:
			tweetText = video.get_title()
			
			sentence = embedding_list(tweetText)
			sentence_in = classifierMixed.prepare_vector(sentence)

			tweet_features = features_tweet(tweetText)
			features_in = classifierMixed.prepare_vector(tweet_features)

			features.append(tweet_classfier_mixnet(sentence_in, features_in).data.tolist()[0][1])

		elif TWEET_CLASSIFIER_DESCRIPTION == feature:
			tweetText = video.get_description()

			sentence = embedding_list(tweetText)
			sentence_in = classifierMixed.prepare_vector(sentence)

			tweet_features = features_tweet(tweetText)
			features_in = classifierMixed.prepare_vector(tweet_features)

			features.append(tweet_classfier_mixnet(sentence_in, features_in).data.tolist()[0][1])

		elif TWEET_CLASSIFIER_COMMENTS == feature:
			#TODO
			pass

	print(features, file=sys.stderr)
	return features

def get_x_y(all_videos, feature_names):
	X = []
	Y = []
	
	for video in all_videos:
		features = get_x(video, feature_names)

		X.append(features)
		Y.append(video.gold_standard)

	
	return X, Y

def get_xs(all_videos, feature_names):
	X = []
	for video in all_videos:
		features = get_x(video, feature_names)
		X.append(features)

	def writeDetailsToFile(output_file, details):
		with open(output_file, 'w') as f:
		    writer = csv.writer(f)
		    writer.writerows(details)

	details = [feature_names]
	details += X

	writeDetailsToFile('feature_values_csv.csv', details)


	return X


def render_dtree(clf,filename, feature_names):
	dot_data = tree.export_graphviz(clf, out_file=None, feature_names = feature_names,
		class_names=["Legit", "Fake"], filled=True)
	graph = graphviz.Source(dot_data)
	graph.render(filename)


def read_list_fake_videos(filename):
	with open(filename) as f:
		videoIds = [x.strip() for x in f.readlines()]
	videos = []
	for videoId in videoIds:
		v = Video(videoId)
		v.gold_standard = 1
		videos.append(v)
	return videos


def dump_classifier(classifier, clf_name):
	fname = os.path.join(MODEL_DIR, clf_name)
	pkl.dump(classifier, open(fname, 'wb'))


def load_classifiers():
	loaded_clfs = []
	for clf_name, clf in clfs:
		loaded_clfs.append((clf_name, pkl.load(open(os.path.join(MODEL_DIR, clf_name),'rb'))))

	return loaded_clfs


def print_evaluation_metrics(Y_true, Y_predicted):
	print(confusion_matrix(Y_true, Y_predicted))
	# print(classification_report(Y_true, Y_predicted))
	print('*'*80)


def main_trial(video_id_file):
	
	try:
		raise FileNotFoundError
		X, Y = pkl.load(open('X_Y_Annotated.pkl', 'rb'))
		print("************** Loaded already built features and labels from pickle file **************")
	except FileNotFoundError:
		annotations = 'VAVD_balanced'
		dataset_all = Dataset(video_id_file,onlyAnnotated=True, annotations=annotations)
		known_videos = [v for v in dataset_all.all_crawled_videos if v.gold_standard!=2]
		known_videos = sorted(known_videos, key=lambda v: v.gold_standard)
		X, Y = get_x_y(known_videos, feature_names)
		pkl.dump((X, Y), open('X_Y_Annotated.pkl', 'wb'))


	num_features = len(X[0])

	# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)#, random_state=2)
	X_train = X_test = X
	Y_train = Y_test = Y

	print("Train on:", len(X_train))
	print("Test on:", len(X_test))

	print("Total number of fake videos in train set: {}".format(sum(Y_train)))
	print("Total number of fake videos in test set: {}".format(sum(Y_test)))

	Y_predicted_all = [1]*len(X_test)
	for clf_name, clf in clfs:
		clf.fit(X_train, Y_train)
		Y_predicted = clf.predict(X_test)
		Y_predicted_all = [Y_predicted[i]*Y_predicted_all[i] for i in range(len(X_test))]
		dump_classifier(clf, clf_name)
		print(clf_name)
		print_evaluation_metrics(Y_test, Y_predicted)

	print("taking_all_into_consideration")
	print_evaluation_metrics(Y_test, Y_predicted_all)


def main_get_annotation_videos(video_id_file):
	
	# dataset_all = Dataset(video_id_file, feature_function=get_x, feature_names=feature_names)
	dataset_all = DatasetScalable(video_id_file)

	clfs = load_classifiers()

	iterator = dataset_all.get_video_and_features(feature_function=get_x, feature_names=feature_names)

	batch_size = 20
	processed = 0

	to_be_annotated = []

	video_to_features = {}

	while True:
		count = 0
		X = []
		video_ids = []
		for video, feature_vector in iterator:
			video_ids.append(video.videoId)
			X.append(feature_vector)
			count += 1

			video_to_features[video.videoId] = feature_vector
			
			if count == batch_size:
				break

		Y = [1]*len(X)
		for clf_name, clf in clfs:
			Y_this = clf.predict(X)
			Y = [Y[i]*Y_this[i] for i in range(len(X))]

		assert(len(video_ids) == len(X) and len(X) == len(Y))

		for i in range(len(video_ids)):
			if Y[i] == 1:
				to_be_annotated.append(video_ids[i])
				print(video_ids[i])

		processed += count

		print("{} processed".format(processed))

		if count != batch_size:
			break

		pkl.dump(video_to_features, open('video_to_features.pkl', 'wb'))

	with open('bootstrap_positive_class_new_ent.txt','w') as final_file:
		final_file.write('\n'.join(to_be_annotated))


def calculate_features(video_id_file):
	dataset_all = Dataset(video_id_file, feature_function=get_x, feature_names=feature_names)
	get_xs(dataset_all.all_crawled_videos, feature_names)


def test_on_fvc(video_id_file):

	annotations = 'fvc_test'
	dataset_all = Dataset(video_id_file,onlyAnnotated=True, annotations=annotations)
	known_videos = [v for v in dataset_all.all_crawled_videos if v.gold_standard!=2]
	known_videos = sorted(known_videos, key=lambda v: v.gold_standard)
	X, Y = get_x_y(known_videos, feature_names)

	clfs = load_classifiers()
	
	Y_predicted_all = [1]*len(X)
	for clf_name, clf in clfs:
		Y_predicted = clf.predict(X)
		Y_predicted_all = [Y_predicted[i]*Y_predicted_all[i] for i in range(len(X))]
		print(clf_name)
		print_evaluation_metrics(Y, Y_predicted)

	print("taking_all_into_consideration")
	print_evaluation_metrics(Y, Y_predicted_all)

		


if __name__ == '__main__':
	# video_id_file = os.path.join(base_dir,'MyVideos','remaining_after_1000_removed.txt')
	# video_id_file = os.path.join(base_dir,'pre_filter_entertainment_2.txt')
	video_id_file = os.path.join(base_dir,'all_videos.txt')


	# calculate_features(video_id_file)
	main_trial(video_id_file)
	# main_get_annotation_videos(video_id_file)

	# exit(0)


	# video_id_file = os.path.join('..', 'YouTube-Spam-Detection','Fake Video Corpus v2.0', 'video_ids.txt')
	test_on_fvc(video_id_file)

