RATIO_VIOLENT_WORDS = "ratio_violent_words"
RATIO_CAPS = "ratio_caps"
HAS_CLICKBAIT_PHRASE = "has_clickbait_phrase"
FIRST_WORD_CAPS = "firstWordCaps"
POSITIVE_SENTIMENT = "POSITIVE_SENTIMENT"
NEGATIVE_SENTIMENT = "NEGATIVE_SENTIMENT"
HAS_QUESTION_MARK = "HAS_QUESTION_MARK"
HAS_EXCLAMATION_MARK = "HAS_EXCLAMATION_MARK"


COMMENTS_FAKENESS = "comments_fakeness"
COMMENTS_CONVERSATION_RATIO = "comments_conversation_ratio"
COMMENTS_INAPPROPRIATENESS = "comments_inappropriateness"
HAS_COMMENTS = "has_comments"

LIKE_RATIO = "likeRatio"
DISLIKE_LIKE_RATIO = "dislike_like_ratio"

TWEET_CLASSIFIER_TITLE = "TWEET_CLASSIFIER_TITLE"
TWEET_CLASSIFIER_DESCRIPTION = "TWEET_CLASSIFIER_DESCRIPTION"
TWEET_CLASSIFIER_COMMENTS = "TWEET_CLASSIFIER_COMMENTS"

LIST_TITLE_FEATURES = [RATIO_VIOLENT_WORDS, RATIO_CAPS, HAS_CLICKBAIT_PHRASE, FIRST_WORD_CAPS, 
						POSITIVE_SENTIMENT, NEGATIVE_SENTIMENT, HAS_QUESTION_MARK, HAS_EXCLAMATION_MARK]

LIST_COMMENTS_FEATURES = [COMMENTS_FAKENESS, COMMENTS_CONVERSATION_RATIO, COMMENTS_INAPPROPRIATENESS, HAS_COMMENTS]
LIST_NON_TEXTUAL_FEATURES = [DISLIKE_LIKE_RATIO]
LIST_TWEET_FEATURES = [TWEET_CLASSIFIER_TITLE, TWEET_CLASSIFIER_DESCRIPTION]#, TWEET_CLASSIFIER_COMMENTS]

LIST_ALL_FEATURES = LIST_TITLE_FEATURES + LIST_COMMENTS_FEATURES + LIST_NON_TEXTUAL_FEATURES
LIST_ALL_AND_TWEET_FEATURES = LIST_ALL_FEATURES + LIST_TWEET_FEATURES

########################################################################################################

LIST_UNCORRELATED = [RATIO_CAPS, RATIO_VIOLENT_WORDS, HAS_CLICKBAIT_PHRASE] + \
					[COMMENTS_FAKENESS, COMMENTS_CONVERSATION_RATIO] + \
					[DISLIKE_LIKE_RATIO] + \
					[TWEET_CLASSIFIER_DESCRIPTION]


########################################################################################################

import os

TWEET_MODEL_PATH = os.path.join('TweetClassifier', 'MixedNet_72.pt')

# MODEL_FILE = os.path.join("models_folder", "decision_tree_march_8.pkl")

MODEL_DIR = 'ClassifierSubsetModels'

base_dir = os.path.join('final_data')