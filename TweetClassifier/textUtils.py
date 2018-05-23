import re
from nltk import word_tokenize
import numpy as np
import os, sys
from gensim.models import KeyedVectors as Word2Vec

# word2vec = None
print("Loading word2vec", file=sys.stderr)
word2vec = Word2Vec.load_word2vec_format('/media/priyank/OS/home/bt1/13CS10037/btp_final_from_server/codes/embeddings/google_news_300.bin', binary=True, limit=500000)
NDIM = 300
print("Loaded word2vec", file=sys.stderr)

def load_word2vec():
	global word2vec, NDIM
	print("Loading word2vec", file=sys.stderr)
	word2vec = Word2Vec.load_word2vec_format('/media/priyank/OS/home/bt1/13CS10037/btp_final_from_server/codes/embeddings/google_news_300.bin', binary=True, limit=500000)
	NDIM = 300
	print("Loaded word2vec", file=sys.stderr)


def replaceSpanishAccent(text):
	text, num_subs_made = re.subn("à|á|â|ä", "a", text)
	text, num_subs_made = re.subn("ò|ó|ô|ö", "o", text)
	text, num_subs_made = re.subn("è|é|ê|ë", "e", text)
	text, num_subs_made = re.subn("ù|ú|û|ü", "u", text)
	text, num_subs_made = re.subn("ì|í|î|ï", "i", text)
	return text

def removeUseless(text):
	text = text.replace("&quot;"," ").replace("&gt;", " ").replace("&lt;", " ").replace("&amp;", " ")
	text = re.sub("http://t.co/[^ ]+ ", "", text)
	text = replaceSpanishAccent(text)
	return text

def removeURL(text):
	text, num_subs_made = re.subn("http://[^ ]+ ", "", text)
	text, num_subs_made = re.subn("https://[^ ]+ ", "", text)
	return text

dir_path = os.path.dirname(os.path.realpath(__file__))

files = {
	"happy_emoticon": open(os.path.join(dir_path, 'resources', 'happy-emoticons.txt')).readlines(),
	"sad_emoticon" : open(os.path.join(dir_path, 'resources', 'sad-emoticons.txt')).readlines(),
	"positive_sentiments": open(os.path.join(dir_path, 'resources', 'positive-words.txt')).readlines(),
	"negative_sentiments": open(os.path.join(dir_path, 'resources', 'negative-words.txt'), encoding="ISO-8859-1").readlines(),
	"first_person_pronouns": open(os.path.join(dir_path, 'resources', 'first-order-prons.txt')).readlines(),
	"second_person_pronouns": open(os.path.join(dir_path, 'resources', 'second-order-prons.txt')).readlines(),
	"third_person_pronouns": open(os.path.join(dir_path, 'resources', 'third-order-prons.txt')).readlines(),
	"slang": open(os.path.join(dir_path, 'resources', 'slangwords-english.txt')).readlines()
}

regpatterns = [r'what the ..ck', r'complete bullshit', r'hate fake', 
				r'give me a ..cking break', r'horse.*shit', r'this is fake', '\bfa+k+e+', 'clickbait'
				r'full of shit',r'bullshit','\bfraud','\bsham','forgery','click bait',
				r'hoax',r'shity video',r'photoshop',r'fake as fuck',r'fake', r'absolute rubbish',
				r'faker',r'nep',r'neuken',r'vol stront',r'wtf',r'falschung',r'not genuine',
				r'counterfeit',r'f.ck','almost looks real','not real']

regexes = [(pat, re.compile(pat, re.IGNORECASE)) for pat in regpatterns]

class TextFeaturesExtractor:

	def __init__(self, text):
		self.originalText = text
		self.uselessRemoved = removeUseless(text)
		self.urlsRemoved = removeURL(self.uselessRemoved)
		self.words = None

	def numURL(self):
		num = 0
		text = self.uselessRemoved
		num += len(re.findall("http://[^ ]+ ", text))
		num += len(re.findall("https://[^ ]+ ", text))
		return num

	def length(self):
		return len(self.urlsRemoved)

	def getFractionUpperCase(self):
		text = self.urlsRemoved
		upper_count = sum(1 for c in text if c.isupper())
		lower_count = sum(1 for c in text if c.islower())
		return (upper_count*1.0)/max((upper_count + lower_count), 1)

	def containsSymbol(self, symbol):
		text = self.urlsRemoved.lower()
		return symbol in text

	def numSymbol(self, symbol):
		text = self.urlsRemoved
		return text.count(symbol)

	def getNumUpperCase(self):
		text = self.urlsRemoved
		return sum(1 for c in text if c.isupper())


	def containsEmo(self, emo_list):
		text = self.urlsRemoved
		for emo in emo_list:
			if emo in text:
				return True
		return False

	def getWords(self):
		if self.words is not None:
			return self.words
		text = self.urlsRemoved
		text = re.sub(r'[^\w]',' ',text)
		tokens = word_tokenize(text)
		tokens = [word.strip(" \r\n\t") for word in tokens]
		return tokens

	def getNumWords(self):
		return len(self.getWords())

	def getNumOccurencesOfFileData(self, filename):
		filedata_list = files[filename]
		count = 0
		for phrase in filedata_list:
			phrase = phrase.strip()
			if phrase in self.urlsRemoved:
				count += 1
		return count


	def getEmbeddingList(self):
		if word2vec is None:
			load_word2vec()
		embeds = []
		for word in self.getWords():
			try:
				embeds.append(word2vec[word.lower()].tolist())
			except KeyError:
				continue
		embeds.append(np.zeros(NDIM).tolist())
		return embeds

	def getEmbedding(self):
		embeds = self.getEmbeddingList()

		if len(embeds) != 0:
			return np.mean(embeds, axis=0).tolist()
		return np.zeros(NDIM).tolist()

	def getFakeness(self):
		for pat, reg in regexes:
			if reg.search(self.urlsRemoved):
				return True
		return False

	def getFakenessVector(self):
		vector = []
		for pat, reg in regexes:
			if reg.search(self.urlsRemoved):
				vector.append(1)
			else:
				vector.append(0)
		return vector
