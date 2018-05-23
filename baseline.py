from sklearn import svm
from TweetClassifier import getTweets
from TweetClassifier.textUtils import TextFeaturesExtractor
from ClassesAndUtil.Dataset import Dataset, DatasetScalable
from ClassesAndUtil.Video import Video
from features import *
from my_constants import *

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split


import os
import pickle as pkl

classifier_pickle = os.path.join('TweetClassifier', 'SVM_ON_TWEETS.pkl')

def train_test_ivc():
    features_and_label = getTweets.getTweetsAndFeatures(os.path.join('TweetClassifier', 'tweet_dataset.tsv'))
    X = []
    Y = []
    for features, label in features_and_label:
        X.append(features)
        Y.append(label)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    clf = svm.SVC(kernel='rbf', probability=True)
    clf.fit(X_train, Y_train)
    Y_predicted = clf.predict(X_test)
    print(confusion_matrix(Y_test, Y_predicted))
    print(classification_report(Y_test, Y_predicted))


def train_level_1():
    features_and_label = getTweets.getTweetsAndFeatures(os.path.join('TweetClassifier', 'tweet_dataset.tsv'))

    X = []
    Y = []
    for features, label in features_and_label:
        X.append(features)
        Y.append(label)

    clf = svm.SVC(kernel='rbf', probability=True)
    clf.fit(X, Y)

    pkl.dump(clf, open(classifier_pickle, 'wb'))

def test_level_1(clf, comment):
    tfe = TextFeaturesExtractor(comment)
    features = getTweets.features_tweet(tfe.urlsRemoved)
    y = clf.predict_proba([features])[0][1]
    return y

def get_comment_features(clf, video):
    bins = 10
    features = [0]*bins
    for comment in video.get_iterator_comments(max_comments=100):
        value = test_level_1(clf, comment.getText())
        features[int(value*bins)-1] += 1
    s = max(sum(features), 1)
    features = [x*1.0/s for x in features]
    
    print(features)
    return features

def load_dataset(video_id_file, annotations):
    dataset_all = Dataset(video_id_file,onlyAnnotated=True, annotations=annotations)
    clf = pkl.load(open(classifier_pickle, 'rb'))
    X = []
    Y = []
    for video in dataset_all.all_crawled_videos:
        features = []
        # features = get_x(video, LIST_TITLE_FEATURES + LIST_NON_TEXTUAL_FEATURES)
        features.extend(get_comment_features(clf, video))
        X.append(features)
        Y.append(video.gold_standard)
    return X, Y

def cross_validate_fvc(X, Y):
    clf = svm.SVC(kernel='rbf')
    # scoring = 'f1'
    scoring = 'f1_macro'
    scores = cross_val_score(clf, X, Y, cv=10, scoring=scoring)

    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def train_test_fvc(X, Y):
    clf = svm.SVC(kernel='rbf')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    clf.fit(X_train, Y_train)
    Y_predicted = clf.predict(X_test)
    print(confusion_matrix(Y_test, Y_predicted))
    print(classification_report(Y_test, Y_predicted))

def train_test_fvc(X_train, Y_train, X_test, Y_test):
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, Y_train)
    Y_predicted = clf.predict(X_test)
    print(confusion_matrix(Y_test, Y_predicted))
    print(classification_report(Y_test, Y_predicted))


if __name__ == '__main__':
    base_dir = os.path.join('final_data')

    # train_test_ivc()
    # exit(0)

    train_level_1()

    # X, Y = load_dataset(os.path.join(base_dir,'all_videos.txt'), 'fvc')
    X_train, Y_train = load_dataset(os.path.join(base_dir,'all_videos.txt'), 'fvc_train')
    X_test, Y_test = load_dataset(os.path.join(base_dir,'all_videos.txt'), 'fvc_test')


    # cross_validate_fvc(X, Y)

    # train_test_fvc(X, Y)
    train_test_fvc(X_train, Y_train, X_test, Y_test)
