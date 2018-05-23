from ClassesAndUtil.Video import Video
import ClassesAndUtil.textProcessor as textProcessor
from my_constants import *
from TweetClassifier.getTweets import features_tweet, embedding_list
from TweetClassifier import classifierMixed

import torch
import sys

def get_x(video, feature_names):

	MAX_COMMENTS = 100

	tweet_classfier_mixnet = classifierMixed.MixNet(300, 300, 2, 308, directions=1)
	tweet_classfier_mixnet.load_state_dict(torch.load(TWEET_MODEL_PATH))

	title = video.get_title()
	sentiment_title = textProcessor.sentimentIntensity(title)

	features = []
	for feature in feature_names:
		if RATIO_VIOLENT_WORDS == feature:
			features.append(textProcessor.ratio_violent_words(title))
		elif RATIO_CAPS == feature:
			features.append(textProcessor.ratio_caps(title))
		elif HAS_CLICKBAIT_PHRASE == feature:
			features.append(int(textProcessor.has_clickbait_phrase(title)))
		elif POSITIVE_SENTIMENT == feature:
			features.append(sentiment_title['pos'])
		elif NEGATIVE_SENTIMENT == feature:
			features.append(sentiment_title['neg'])
		elif HAS_QUESTION_MARK == feature:
			features.append(int('?' in title))
		elif HAS_EXCLAMATION_MARK == feature:
			features.append(int('!' in title))


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

	# print(features, file=sys.stderr)
	return features

def get_x_y(all_videos, feature_names):
	X = []
	Y = []
	
	for video in all_videos:
		features = get_x(video, feature_names)
		X.append(features)
		Y.append(video.gold_standard)

	return X, Y

def get_x_baseline(video):
	MAX_COMMENTS = 100
	features = []

	channel = video.get_channel()
	features.append(channel.getDetail("viewCount"))
	features.append(channel.getDetail("commentCount"))
	features.append(channel.getDetail("subscriberCount"))
	features.append(channel.getDetail("videoCount"))
	
