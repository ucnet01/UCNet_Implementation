import os
import json
import isodate
from abc import ABCMeta

from ClassesAndUtil import textProcessor

# base_dir = os.path.join('..','YouTube-Spam-Detection', 'Entdata')
# base_dir = os.path.join('..','YouTube-Spam-Detection','data')
# base_dir = os.path.join('..','YouTube-Spam-Detection','Fake Video Corpus v2.0')
base_dir = os.path.join('final_data')


videoFolder = os.path.join(base_dir, 'VideoDetailsWithLive')# "../data/VideoDetails"
channelFolder = os.path.join(base_dir, 'ChannelDetails')# "../data/ChannelDetails/"
commentsFolder = os.path.join(base_dir, 'CommentsandRepliesNew')# "../data/CommentsandRepliesNew/"


class Video:

	def __init__(self, videoId):
		self.videoId = videoId
		self.videoFile = self.__getVideoFile(videoId)
		self.channelFile = self.__getChannelFile(videoId)
		self.commentsFile = self.__getCommentsFile(videoId)
		
		self.legitimate = ((self.videoFile is not None) and (self.channelFile is not None) and (self.commentsFile is not None))

		self.videoDetails = None
		self.channel = None
		self.comments = None

		if self.legitimate:
			if not self.get_title():
				self.legitimate = False
			else:
				self.channel = self.__getChannel()

		# self.__create_comments()
		# if self.comments and len(self.comments) != 0:
		# 	print self.comments[0].getText()

	def __getVideoFile(self, videoId):
		filename = os.path.join(videoFolder, "Video_{id}.json".format(id=videoId.strip()))
		# filename = "{Folder}/Video_{videoId}.json".format(Folder=videoFolder, videoId=videoId.strip())
		if os.path.isfile(filename) == False:
			return None
		return filename

	def __getChannelFile(self, videoId):
		filename = os.path.join(channelFolder, "Video_{id}.json".format(id=videoId.strip()))
		# filename = "{Folder}/Video_{videoId}.json".format(Folder=channelFolder, videoId=videoId.strip())
		if os.path.isfile(filename) == False:
			return None
		return filename

	def __getCommentsFile(self, videoId):
		filename = os.path.join(commentsFolder, "Video_{id}.json".format(id=videoId.strip()))
		# filename = "{Folder}/Video_{videoId}.json".format(Folder=commentsFolder, videoId=videoId.strip())
		if os.path.isfile(filename) == False:
			return None
		return filename

	def __getChannel(self):
		if self.channelFile is None:
			return None
		with open(self.videoFile) as f:
			videojson = json.load(f)
		channelId = videojson["items"][0]["snippet"]["channelId"].strip()#.encode("utf-8")
		return Channel.get_or_create_channel(channelId, self.channelFile)

	def __create_comments(self):
		if self.commentsFile is None:
			return None

		with open(self.commentsFile) as f:
			comments_json = json.load(f)

		assert len(comments_json) == 1
		comments_json = list(comments_json.values())[0]

		self.comments = []

		for comment_id in comments_json:
			comment = TopLevelComment(self.videoId, comment_id, comments_json[comment_id])
			if comment.commentProperties is not None:	# Comment not available now! Maybe deleted :/
				self.comments.append(comment)
				assert self.videoId == comment.videoId

	def get_iterator_comments(self, max_comments=100000):
		if self.comments is None:
			self.__create_comments()
		i = 0
		for cmt in self.comments:
			yield cmt
			i += 1
			if i >= max_comments:
				break

	def get_num_comments(self):
		if self.comments is None:
			self.__create_comments()
		return len(self.comments)

	def get_comments_inappropriateness(self, max_comments=100000):
		total_inappropriateness = 0
		denominator = 0
		for comment in self.get_iterator_comments(max_comments=max_comments):
			total_inappropriateness += comment.get_inappropriateness()
			denominator += 1
		if denominator == 0:
			return 0.0
		return total_inappropriateness/denominator

	def get_comments_fakeness(self, max_comments=100000, for_zero_comments=0.0):
		"""
			Fakeness Indicating Comments
		"""
		numerator = sum([cmt.says_fake() for cmt in self.get_iterator_comments(max_comments=max_comments)])
		denominator = min(len(self.comments), max_comments)
		if denominator == 0:
			return for_zero_comments
		return (numerator*1.0)/denominator

	
	def get_title(self):
		try:
			if self.videoDetails is None:
				with open(self.videoFile) as f:
					self.videoDetails = json.load(f)["items"][0]
			return self.videoDetails["snippet"]["title"]
		except Exception:
			return None

	def get_conversation_ratio(self, recursive=True, max_comments=100000):
		"""
			Conversation Ratio in Comments
		"""
		total_conversational_comments = 0
		denominator = 0
		for comment in self.get_iterator_comments(max_comments=max_comments):
			if recursive:
				addThis = comment.isConversationalRecursive()
			else:
				addThis = comment.isConversational()

			total_conversational_comments += addThis
			if addThis > 0:
				denominator += addThis
			elif addThis == 0:
				denominator += 1
			else:
				raise Exception
		if denominator == 0:
			return 0.0
		return (total_conversational_comments * 1.0) / denominator

	def get_like_count(self):
		if self.videoDetails is None:
			with open(self.videoFile) as f:
				self.videoDetails = json.load(f)["items"][0]
		try:
			return int(self.videoDetails["statistics"]["likeCount"])
		except KeyError:
			return 0

	def get_dislike_count(self):
		if self.videoDetails is None:
			with open(self.videoFile) as f:
				self.videoDetails = json.load(f)["items"][0]
		try:
			return int(self.videoDetails["statistics"]["dislikeCount"])
		except KeyError:
			return 0

	def get_tags(self):
		if self.videoDetails is None:
			with open(self.videoFile) as f:
				self.videoDetails = json.load(f)["items"][0]
		try:
			return self.videoDetails["snippet"]["tags"]
		except KeyError:
			return []
		
	def get_description(self):
		if self.videoDetails is None:
			with open(self.videoFile) as f:
				self.videoDetails = json.load(f)["items"][0]
		return self.videoDetails["snippet"]["description"].strip()
	
	def get_duration_in_timedelta(self):
		if self.videoDetails is None:
			with open(self.videoFile) as f:
				self.videoDetails = json.load(f)["items"][0]
		return isodate.parse_duration(self.videoDetails["contentDetails"]["duration"])	

	def get_default_audio_language(self, if_not_provided="en"):
		if self.videoDetails is None:
			with open(self.videoFile) as f:
				self.videoDetails = json.load(f)["items"][0]
		return self.videoDetails["snippet"].get("defaultAudioLanguage", if_not_provided)

	def get_default_language(self, if_not_provided="en"):
		if self.videoDetails is None:
			with open(self.videoFile) as f:
				self.videoDetails = json.load(f)["items"][0]
		return self.videoDetails["snippet"].get("defaultLanguage", if_not_provided)

	def is_live(self):
		if self.videoDetails is None:
			with open(self.videoFile) as f:
				self.videoDetails = json.load(f)["items"][0]
		if self.videoDetails.get("liveStreamingDetails") is not None:
			return True
		return False

	def get_category(self):
		if self.videoDetails is None:
			with open(self.videoFile) as f:
				self.videoDetails = json.load(f)["items"][0]
		return self.videoDetails["snippet"]["categoryId"]

	def get_channel(self):
		if self.channel is None:
			self.channel = self.__getChannel()
		return self.channel





class Channel:

	loadedChannels = {}

	@staticmethod
	def get_or_create_channel(channelId, channelFile):
		channel = Channel.loadedChannels.get(channelId)
		if channel:
			return channel

		if channelFile is None:
			# This condition could arrive while creating channel graph when we don't have any video for this channel
			channelFile = os.path.join('..','..','data','ChannelByIdNew','{}.json'.format(channelId))

		if not os.path.isfile(channelFile):
			raise FileNotFoundError
		
		Channel.loadedChannels[channelId] = Channel(channelId, channelFile)
		return Channel.loadedChannels[channelId]

	def __init__(self, channelId, channelFile):
		self.channelId = channelId
		self.channelFile = channelFile
		self.channelDetails = None
	
	def getDetail(self, key):
		if self.channelDetails is None:
			self.readChannelFeatures()
		return self.channelDetails[key]


	def readChannelFeatures(self):
		self.channelDetails = {}
		channelDetailsOriginal = {}
		
		with open(self.channelFile) as f:
			channelDetailsOriginal = json.load(f)

		if channelDetailsOriginal.get("items") is not None:
			channelDetailsOriginal = channelDetailsOriginal["items"][0]
		
		self.channelDetails["videoCount"] = int(channelDetailsOriginal["statistics"]["videoCount"])
		self.channelDetails["commentCount"] = int(channelDetailsOriginal["statistics"]["commentCount"])
		self.channelDetails["viewCount"] = int(channelDetailsOriginal["statistics"]["viewCount"])
		self.channelDetails["subscriberCount"] = int(channelDetailsOriginal["statistics"]["subscriberCount"])
		self.channelDetails["brandingSettingsImages"] = channelDetailsOriginal["brandingSettings"]["image"]

		self.channelDetails["description"] = channelDetailsOriginal["snippet"]["description"]
		self.channelDetails["title"] = channelDetailsOriginal["snippet"]["title"]
		
		try:
			self.channelDetails["keywords"] = channelDetailsOriginal["brandingSettings"]["channel"]["keywords"]
		except KeyError:
			self.channelDetails["keywords"] = "NONE"
		try:
			self.channelDetails["featuredChannels"] = channelDetailsOriginal["brandingSettings"]["channel"]["featuredChannelsUrls"]
			assert type(self.channelDetails["featuredChannels"]) == type([])
		except KeyError:
			self.channelDetails["featuredChannels"] = []
		self.channelDetails["numFeaturedChannels"] = len(self.channelDetails["featuredChannels"])
		
		try:
			videoCount = max(int(channelDetailsOriginal["statistics"]["videoCount"]),1)
			self.channelDetails["commentCountRatio"] = float(int(channelDetailsOriginal["statistics"]["commentCount"]))  / float(videoCount)
			self.channelDetails["viewCountRatio"] = float(int(channelDetailsOriginal["statistics"]["viewCount"]))  / float(videoCount)
			self.channelDetails["subscriberCountRatiovideo"] = float(int(channelDetailsOriginal["statistics"]["subscriberCount"]))  / float(videoCount)
			self.channelDetails["subscriberCountRatioviews"] = float(int(channelDetailsOriginal["statistics"]["subscriberCount"]))  / float(max(int(channelDetailsOriginal["statistics"]["viewCount"]),1))
		except Exception as e:
			print("exception in Channel:",e)


class Comment:
	"""Abstract Class"""
	__metaclass__ = ABCMeta
	
	def __init__(self, videoId):
		self.videoId = videoId

	def getText(self):
		return self.commentProperties["textOriginal"]

	def getLikeCount(self):
		return self.commentProperties["likeCount"]

	def get_inappropriateness(self):
		return textProcessor.inappropriateness(self.getText())

	def says_fake(self):
		return textProcessor.fakeness(self.getText())

	def fakeness_vector(self):
		return textProcessor.fakeness_vector(self.getText())
	
	def isConversational(self):
		if isinstance(self, TopLevelComment):
			return len(self.replies) > 0
		else:
			return True	

class TopLevelComment(Comment):

	def __init__(self, videoId, commentId, comment_values):
		try:
			super(TopLevelComment, self).__init__(videoId)
			self.commentId = commentId
			if comment_values["main_comment"] == "":
				# print videoId
				self.commentProperties = None
				self.replies = None
				return
			self.commentProperties = comment_values["main_comment"]["topLevelComment"]["snippet"]
			self.replies = [RepliedComment(videoId, commentId, reply["snippet"]) for reply in comment_values["replied_comment"]]
		except Exception as e:
			print(e)
			print(commentId)
			print(json.dumps(comment_values, indent=4))
			exit(0)

	def isConversationalRecursive(self):
		if len(self.replies) > 0:
			return len(self.replies) + 1
		return 0
		


class RepliedComment(Comment):

	def __init__(self, videoId, parentId, comment_values):
		super(RepliedComment, self).__init__(videoId)
		self.parentId = parentId
		self.commentProperties = comment_values
