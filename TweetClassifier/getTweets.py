import csv
from TweetClassifier.textUtils import TextFeaturesExtractor
from TweetClassifier.constantsTweet import *
# from textUtils import TextFeaturesExtractor
# from constantsTweet import *

def iterate_tweets(filename):
	with open(filename) as csvFile:
		reader = csvFile.read().split('\n')
	reader.pop(0)	## To remove the header
	for row in reader:
		row = row.split('\t')
		tweet = row[1].strip()
		labelText = row[-1].strip().lower()
		assert(labelText == 'fake' or labelText == 'real')
		label = 0
		if labelText == 'fake':
			label = 1
		yield tweet, label


def features_tweet(tweetText, features_to_be_extracted=ALL_FEATURES):
	features = []
	tfe = TextFeaturesExtractor(tweetText)
	for feature_name in features_to_be_extracted:
		if feature_name == LENGTH:
			features.append(tfe.length())
		elif feature_name == NUM_WORDS:
			features.append(tfe.getNumWords())
		elif feature_name == CONTAINS_QUESTION_MARK:
			features.append(tfe.containsSymbol('?'))
		elif feature_name == CONTAINS_EXCLAMATION_MARK:
			features.append(tfe.containsSymbol('!'))
		
		elif feature_name == NUM_QUESTION_MARK:
			features.append(tfe.numSymbol('?'))
		elif feature_name == NUM_EXCLAMATION_MARK:
			features.append(tfe.numSymbol('!'))
		
		elif feature_name == CONTAINS_HAPPY_EMOTICON:
			features.append(tfe.getNumOccurencesOfFileData("happy_emoticon"))
		elif feature_name == CONTAINS_SAD_EMOTICON:
			features.append(tfe.getNumOccurencesOfFileData("sad_emoticon"))
		
		elif feature_name == NUM_UPPER_CASE:
			features.append(tfe.getNumUpperCase())
		elif feature_name == CONTAINS_COLON:
			features.append(tfe.containsSymbol(':'))
		elif feature_name == CONTAINS_PLEASE:
			features.append(tfe.containsSymbol('please'))
		elif feature_name == NUM_POSITIVE_SENTIMENTS:
			features.append(tfe.getNumOccurencesOfFileData("positive_sentiments"))
		elif feature_name == NUM_NEGATIVE_SENTIMENTS:
			features.append(tfe.getNumOccurencesOfFileData("negative_sentiments"))
		elif feature_name == NUM_FIRST_PERSON_PRON:
			features.append(tfe.getNumOccurencesOfFileData("first_person_pronouns"))
		elif feature_name == NUM_SECOND_PERSON_PRON:
			features.append(tfe.getNumOccurencesOfFileData("second_person_pronouns"))
		elif feature_name == NUM_THIRD_PERSON_PRON:
			features.append(tfe.getNumOccurencesOfFileData("third_person_pronouns"))
		elif feature_name == NUM_SLANG:
			features.append(tfe.getNumOccurencesOfFileData("slang"))
		
		elif feature_name == FRACTION_UPPER_CASE:
			features.append(tfe.getFractionUpperCase())
		elif feature_name == NUM_URL:
			features.append(tfe.numURL())
		elif feature_name == EMBEDDING:
			features += tfe.getEmbedding()
	return features

def embedding_list(tweetText):
	tfe = TextFeaturesExtractor(tweetText)
	return tfe.getEmbeddingList()


def getTweetsAndFeatures(tweetFileName, features_to_be_extracted=ALL_FEATURES):
	tweetIterator = iterate_tweets(tweetFileName)
	dataset = []
	for tweet, label in tweetIterator:
		dataset.append((features_tweet(tweet, features_to_be_extracted), label))
		# print(features_tweet(tweet, features_to_be_extracted))
		# print(label, tweet)
	return dataset

def getTweetEmbeddingLists(tweetFileName):
	tweetIterator = iterate_tweets(tweetFileName)
	dataset = []
	for tweet, label in tweetIterator:
		dataset.append((embedding_list(tweet), label))
	return dataset

def getTweetsEmbeddingsAndFeatures(tweetFileName, features_to_be_extracted=ALL_FEATURES):
	tweetIterator = iterate_tweets(tweetFileName)
	dataset = []
	for tweet, label in tweetIterator:
		dataset.append((embedding_list(tweet), features_tweet(tweet, features_to_be_extracted), label))
	return dataset

if __name__ == '__main__':
	getTweetsAndFeatures('tweet_dataset.tsv', ALL_FEATURES)