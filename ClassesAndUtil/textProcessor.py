import sys
import nltk
import re
import pprint
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop = stopwords.words('english')
lemma=WordNetLemmatizer()
sentiment_intensity_analyzer = SentimentIntensityAnalyzer()
sentiment_names = ["compound", "neg", "neu", "pos"]

#Load Swear Words
Swear_Words_File = "./BOW_Files/swear_final_lemma.txt"
with open(Swear_Words_File) as f:
	swear_words=f.readlines()
swear_words=set([words.strip(" \r\n\t").lower() for words in swear_words])

#Load fakeness regexes
Fake_Regex_File = "./BOW_Files/fakeness_words.txt"
with open(Fake_Regex_File) as f:
	fake_regexes=f.readlines()
fake_regexes=[re.compile(words.strip(" \r\n\t").lower()) for words in fake_regexes]

#Load Clickbait phrases
Clickbait_Phrases_File = "./BOW_Files/clickbait_phrases.txt"
with open(Clickbait_Phrases_File) as f:
	clickbait_phrases=f.readlines()
clickbait_phrases=[phrase.strip(" \r\n\t").lower() for phrase in clickbait_phrases]

#Load Violent Words
Violent_Words_File = "./BOW_Files/violent_words.txt"
with open(Violent_Words_File) as f:
	violent_words=f.readlines()
violent_words=set([lemma.lemmatize(words.strip(" \r\n\t").lower()) for words in violent_words])

#Load 2 letter words
two_letter_words_file = "./BOW_Files/two_letter_words.txt"
with open(two_letter_words_file) as f:
	two_letter_words=[x.split('\t')[0].strip() for x in f.readlines()]
two_letter_words=set([words.strip(" \r\n\t").lower() for words in violent_words])

#Load Known Common Abbreviations
Common_Abbreviations_File = "./BOW_Files/common_abbreviations.txt"
with open(Common_Abbreviations_File) as f:
	common_abbreviations=f.readlines()
common_abbreviations=set([words.strip(" \r\n\t").lower() for words in common_abbreviations])

def words_lowercase(text):
	text = re.sub(r'[^\w]',' ',text)
	tokens = word_tokenize(text)
	tokens = [word.lower().strip(" \r\n\t") for word in tokens]
	return tokens


def words_originalcase(text):
	text = re.sub(r'[^\w]',' ',text)
	tokens = word_tokenize(text)
	tokens = [word.strip(" \r\n\t") for word in tokens]
	return tokens


def inappropriateness(text,binary=True):
	global swear_words
	count_sword=0
	count_total=0
	tokens = words_lowercase(text)
	
	if len(tokens) > 0:
		count_total = 2*len(tokens) - 1 	# Unigrams and Bigrams both
	
	for word in tokens:
		if len(word) == 1 or word in stop:
			continue
		word = lemma.lemmatize(word)
		if word in swear_words:
			count_sword += 1
			if binary:
				return 1.0
	
	for i in range(len(tokens) - 1):
		word_curr = tokens[i]
		word_next = tokens[i+1]
		if len(word_curr) == 1 or word_curr in stop:
			continue
		word_bi=word_curr + word_next
		if word_bi in swear_words:
			count_sword += 1
			if binary:
				return 1.0
	
	if count_total == 0 or binary:
		return 0.0
	return (count_sword*100.0)/count_total



regpatterns = [r'what the ..ck', r'what the ..ck is this', r'complete bullshit',r'hate fake', 
			r'give me a ..cking break', r'horse.*shit', r'this is fake', '\bfa+k+e+', 'clickbait'
			r'full of shit',r'bullshit',r'fake as can be','\bfraud','\bsham','forgery','click bait',
			r'hoax',r'shity video',r'photoshop',r'fake as fuck',r'fake', r'absolute rubbish',
			r'faker',r'more fake',r'nep',r'neuken',r'vol stront',r'wtf',r'falschung',r'not genuine',
			r'counterfeit',r'f.ck','fake bitch','almost looks real', 'fake fake','not real']

regexes = [(pat, re.compile(pat, re.IGNORECASE)) for pat in regpatterns]

def fakeness(text):
	
	for pat, reg in regexes:
		if reg.search(text):
			# print(pat, ":", text, file=sys.stderr)
			return True
	return False

	# tokens = words_lowercase(text)
	# for word in tokens:
	# 	for fake_regex in fake_regexes:
	# 		if fake_regex.search(word):
	# 			# print fake_regex.pattern, word
	# 			return True
	# return False

def fakeness_vector(text):
	vector = []
	for pat, reg in regexes:
		if reg.search(text):
			vector.append(1)
		else:
			vector.append(0)
	return vector

def sentimentIntensity(text):
	sentiments = sentiment_intensity_analyzer.polarity_scores(text)
	return sentiments

def pos_tagged(text):
	tokenized = words_lowercase(text)
	tagged = pos_tag(tokenized)
	return tagged

def num_superlative(text):
	tagged = pos_tagged(text)
	words, tags = zip(*tagged)
	count = 0
	for tag in tags:
		if tag == 'JJS':
			count += 1
	return count

def num_adjectives(text):
	tagged = pos_tagged(text)
	words, tags = zip(*tagged)
	count = 0
	for tag in tags:
		if 'JJ' in tag:
			count += 1
	return count

def has_clickbait_phrase(text):
	text = text.lower()
	
	for phrase in clickbait_phrases:
		if phrase in text:
			return True
	return False


def ratio_caps(text):
	words = words_originalcase(text)
	count = 0
	denominator = 0
	for word in words:
		if isAbbreviation(word):
			continue
		elif word.isupper():
			count += 1
		denominator += 1

	return count*1.0/max(denominator, 1)


def ratio_violent_words(text):
	tokens_and_pos = pos_tagged(text)
	lemmatized = set()
	for token, pos in tokens_and_pos:
		if pos.lower()[0] == 'v':
			lemmatized.add(lemma.lemmatize(token, 'v'))
		else:
			lemmatized.add(lemma.lemmatize(token, 'n'))
	return len(lemmatized & violent_words)*1.0/len(lemmatized)


def isAbbreviation(word):
	if word.upper() != word:
		return False
	if '.' in word:
		return True
	if len(set('aeiouy').intersection(word.lower())) == 0:
		return True
	if len(word) <= 2 and word.lower() not in two_letter_words:
		return True
	if word.lower() in common_abbreviations:
		return True
	return False

def first_word_capital(text):
	word = words_originalcase(text)[0]
	if isAbbreviation(word):
		return False
	return word.isupper()