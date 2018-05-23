from ClassesAndUtil.Video import Video, Channel

import os, sys
import pickle as pkl
import json

# base_dir1 = os.path.join('..','YouTube-Spam-Detection','data')
# base_dir1 = os.path.join('..','YouTube-Spam-Detection','Fake Video Corpus v2.0')
base_dir1 = os.path.join('final_data')

with open(os.path.join(base_dir1, 'annotations_train.json')) as f:
	annotated_train = json.load(f)
with open(os.path.join(base_dir1, 'annotations_fvc.json')) as f:
	annotated_fvc = json.load(f)

annotated = {}
for key in annotated_train:
	if annotated_train[key] != 2:
		annotated[key] = annotated_train[key]
for key in annotated_fvc:
	annotated[key] = annotated_fvc[key]


class Dataset:
	
	def __init__(self, id_file, feature_function=None, feature_names=None, annotations=None, load_saved=False, maxVideos=None, onlyAnnotated=False, base_dir=base_dir1):
		global annotated

		self.all_crawled_videos = []
		self.filtered_videos = []
		self.feature_function = feature_function
		self.feature_names = feature_names

		with open(os.path.join(base_dir1, 'annotations_{}.json'.format(annotations))) as f:
			annotated = json.load(f)

		# if annotations == 'train':
		# 	annotated = annotated_train
		# elif annotations == 'fvc':
		# 	annotated = annotated_fvc

		print("No. of annotated videos:", len(annotated), file=sys.stderr)

		if load_saved:
			self.all_crawled_videos = pkl.load(open('dataset_all.pkl', 'rb'))

		else:
			with open(id_file) as f:
				video_ids = [x.strip() for x in f.readlines()]
			print(len(video_ids))
			
			if onlyAnnotated:
				video_ids = list(set(video_ids) & set(annotated.keys()))
			
			if maxVideos is not None:
				video_ids = video_ids[:maxVideos]
			print("len(video_ids): {}".format(len(video_ids)), file=sys.stderr)
			
			i = 0
			fake = 0
			for video_id in video_ids:
				v = Video(video_id)
				if v.legitimate and onlyAnnotated:
					v.gold_standard = annotated[video_id]
					if v.gold_standard == 1:
						fake+=1
						print(v.videoId, v.get_title(), file=sys.stderr)
				if v.legitimate:
					self.all_crawled_videos.append(v)
				

			pkl.dump(self.all_crawled_videos, open('dataset_all.pkl', 'wb'))
		
		print('All legitimate crawled videos: {}'.format(len(self.all_crawled_videos)), file=sys.stderr)
		print('Fake: {}'.format(fake), file=sys.stderr)

	def get_video_and_features(self):
		for video in self.all_crawled_videos:
			yield video, self.feature_function(video, self.feature_names)


class DatasetScalable:

	def __init__(self, id_file, annotations=None, onlyAnnotated=False, base_dir=base_dir1, maxVideos=None):
		global annotated
		self.onlyAnnotated = onlyAnnotated

		with open(os.path.join(base_dir1, 'annotations_{}.json'.format(annotations))) as f:
			annotated = json.load(f)

		# if annotations == 'train':
		# 	annotated = annotated_train
		# elif annotations == 'fvc':
		# 	annotated = annotated_fvc
		
		with open(id_file) as f:
			video_ids = [x.strip() for x in f.readlines()]
			
		if onlyAnnotated:
			video_ids = list(set(video_ids) & set(annotated.keys()))
		
		if maxVideos is not None:
			video_ids = video_ids[:maxVideos]

		self.video_ids = video_ids
		print("len(video_ids): {}".format(len(self.video_ids)), file=sys.stderr)

	def get_video_and_features(self, feature_function, feature_names):

		i = 0
		for video_id in self.video_ids:
			v = Video(video_id)
			if v.legitimate and self.onlyAnnotated:
				v.gold_standard = annotated[video_id]
				if v.gold_standard == 1:
					print(v.videoId, v.get_title(), file=sys.stderr)
			if v.legitimate:
				yield v, feature_function(v, feature_names)
				# self.all_crawled_videos.append(v)\

	def get_videos(self):
		i = 0
		for video_id in self.video_ids:
			v = Video(video_id)
			if v.legitimate and self.onlyAnnotated:
				v.gold_standard = annotated[video_id]
				if v.gold_standard == 1:
					print(v.videoId, v.get_title(), file=sys.stderr)
			if v.legitimate:
				yield v

