from ClassesAndUtil.Video import Video, Channel

import os
import pickle as pkl
import json

with open('../data/annotateResult_final.json') as f:
	annotated = json.load(f)

class Dataset:
	
	def __init__(self, id_file, feature_function=None, feature_names=None, load_saved=False, maxVideos=None, onlyAnnotated=False):
		self.all_crawled_videos = []
		self.filtered_videos = []
		self.feature_function = feature_function
		self.feature_names = feature_names

		if load_saved:
			self.all_crawled_videos = pkl.load(open('dataset_all.pkl', 'rb'))

		else:
			with open(id_file) as f:
				video_ids = [x.strip() for x in f.readlines()]
			
			if onlyAnnotated:
				video_ids = list(set(video_ids) & set(annotated.keys()))
			
			if maxVideos is not None:
				video_ids = video_ids[:maxVideos]
			print("len(video_ids): {}".format(len(video_ids)))
			
			i = 0
			for video_id in video_ids:
				v = Video(video_id)
				if v.legitimate and onlyAnnotated:
					v.gold_standard = annotated[video_id]
					if v.gold_standard == 1:
						print(v.videoId, v.get_title())
				if v.legitimate:
					self.all_crawled_videos.append(v)
				

			pkl.dump(self.all_crawled_videos, open('dataset_all.pkl', 'wb'))
		
		print('All legitimate crawled videos: {}'.format(len(self.all_crawled_videos)))


	def get_video_and_features(self):
		for video in self.all_crawled_videos:
			yield video, self.feature_function(video, self.feature_names)
