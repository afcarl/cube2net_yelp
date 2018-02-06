import pickle
import sys
import os
import re
import json
import random
import time
from collections import defaultdict
from subprocess import call

class YelpCube(object):
	def __init__(self, params):
		self.params = params

		self.business_id = {}
		self.user_id = {}
		self.business_user = []
		self.user_business = []
		self.review_business = []

		self.category_name = []
		self.category_business = []
		self.city_name = []
		self.city_business = []
		self.topic_name = [[] for x in range(self.params['num_topics'])]
		self.topic_business = [set() for x in range(self.params['num_topics'])]

	def step1(self):
		#input businesses
		num_business = 0
		valid_business = 0
		with open(self.params['yelp_business'], 'r') as f:
			for line in f:
				p = json.loads(line)
				num_business += 1
				if ('business_id' not in p) \
				or ('city' not in p) \
				or ('state' not in p) \
				or ('review_count' not in p) \
				or (int(p['review_count']) < 10) \
				or ('attributes' not in p) \
				or ('categories' not in p):
					continue

				self.business_id[p['business_id']] = valid_business
				valid_business += 1

				for cat in map(lambda x: x.lower(), p['categories']):
					if cat not in self.category_name:
						self.category_name.append(cat)
						self.category_business.append(set())
					self.category_business[self.category_name.index(cat)].add(valid_business-1)
				if p['city'] not in self.city_name:
					self.city_name.append(p['city'])
					self.city_business.append(set())
				self.city_business[self.city_name.index(p['city'])].add(valid_business-1)

		self.business_user = [set() for i in range(len(self.business_id))]
		print('finised input businesses: %d/%d' % (valid_business, num_business) )

		#input users
		num_user = 0
		valid_user = 0
		with open(self.params['yelp_user'], 'r') as f:
			for line in f:
				p = json.loads(line)
				num_user += 1
				if ('user_id' not in p) \
				or ('review_count' not in p) \
				or (int(p['review_count']) < 10):
					continue

				self.user_id[p['user_id']] = valid_user
				valid_user += 1

		self.user_business = [set() for i in range(len(self.user_id))]
		print('finised input users: %d/%d' % (valid_user, num_user) )

		#input reviews
		num_review = 0
		with open(self.params['yelp_review'], 'r') as f, open(self.params['content_file'], 'w') as cf:
			for line in f:
				p = json.loads(line)
				num_review += 1
				if ('business_id' not in p) \
				or ('user_id' not in p) \
				or ('text' not in p) \
				or (not re.match("^[\w\s,.:?-]+$", p['text'])):
					continue
			
				try:
					bid = self.business_id[p['business_id']]
					uid = self.user_id[p['user_id']]
				except KeyError:
					continue

				cf.write((p['text'].replace('\n', '').replace('\r', '')+'\n').encode('utf-8'))
				self.review_business.append(bid)
				self.business_user[bid].add(uid)
				self.user_business[uid].add(bid)

		print('finised input reviews: %d/%d' % (len(self.review_business), num_review) )

		'''
		#input checkins
		num_checkin = 0
		valid_checkin = 0
		with open(self.params['yelp_checkin'], 'r') as f:
			for line in f:
				p = json.loads(line)
				num_checkin += 1
				if ('business_id' not in p) \
				or (p['business_id'] not in self.business) \
				or ('user_id' not in p) \
				or (p['user_id'] not in self.user):
					continue

				valid_checkin += 1
				if num_checkin % 1000 == 0:
					print('proccessed %d/%d checkins' % (valid_checkin, num_checkin))
				#self.business_user[self.business.index(p['business_id'])].add(self.user.index(p['user_id']))
				#self.user_business[self.user.index(p['user_id'])].add(self.business.index(p['business_id']))

		print('finised input checkins: %d/%d' % (valid_checkin, num_checkin) )
		'''
		
		with open('models/step1.pkl', 'wb') as f:
			pickle.dump(self, f)

		print('step1: finished.')

	def step2(self):
		#sample businesses
		if not os.path.exists('models/basenet.pkl'):
			print('generating basenet.')
			basenet = {}
			basenet['set0_business'] = set()
			basenet['set0_link'] = set()
			basenet['set1_business'] = set()
			basenet['set1_link'] = set()

			business_ca_takeout = set()
			business_il_goodforkids = set()

			with open(self.params['yelp_business'], 'r') as f:
				for line in f:
					p = json.loads(line)
					if ('business_id' not in p) \
					or (p['business_id'] not in self.business_id):
						continue

				if p['state'].lower() == 'ca' and 'RestaurantsTakeOut' in p['attributes'] and p['attributes']['RestaurantsTakeOut']:
					business_ca_takeout.add(valid_business-1)
				if p['state'].lower() == 'il' and 'GoodForKids' in p['attributes'] and p['attributes']['GoodForKids']:
					business_il_goodforkids.add(valid_business-1)


			print('got %d ca_takeout business, %d il_goodforkids business' % (len(business_ca_takeout), len(business_il_goodforkids)))

			for b in self.business_ca_takeout:
				if random.random() < 0.5:
					basenet['set0_business'].add(b)
					for u in self.business_user[b]:
						if random.random() < 0.2:
							basenet['set0_link'].add((b, u))

			print('generated basenet 0 with %d business and %d held links.' %(len(basenet['set0_business']), len(basenet['set0_link'])))

			for b in self.business_il_goodforkids:
				if random.random() < 0.5:
					basenet['set1_business'].add(b)
					for u in self.business_user[b]:
						if random.random() < 0.2:
							basenet['set1_link'].add((b, u))

			print('generated basenet 1 with %d business and %d held links.' %(len(basenet['set1_business']), len(basenet['set1_link'])))
			
			with open('models/basenet.pkl', 'wb') as f:
				pickle.dump(basenet, f)

		if not os.path.exists('models/segmentation.txt'):
			call('./phrasal_segmentation.sh', shell=True, cwd='../AutoPhrase')
		texts = []
		line_num = 0
		content = []
		tag_beg = '<phrase>'
		tag_end = '</phrase>'
		with open('models/segmentation.txt', 'r') as f:
			for line in f:
				while line.find(tag_beg) >= 0:
					beg = line.find(tag_beg)
					end = line.find(tag_end)+len(tag_end)
					content.append(line[beg:end].replace(tag_beg, '').replace(tag_end, '').lower())
					line = line[:beg] + line[end:]
					texts.append(content)
					content = []

		print("lda: constructing dictionary")
		dictionary = corpora.Dictionary(texts)
		print("lda: constructing doc-phrase matrix")
		corpus = [dictionary.doc2bow(text) for text in texts]
		print("lda: computing model")
		if not os.path.exists('models/ldamodel.pkl'):
			ldamodel = models.ldamodel.LdaModel(corpus, num_topics=self.params['num_topics'], id2word = dictionary, passes=20)
			with open('models/ldamodel.pkl', 'wb') as f:
				pickle.dump(ldamodel, f)
		else:
			with open('models/ldamodel.pkl', 'rb') as f:
				ldamodel = pickle.load(f)
		print("lda: saving topical phrases")
		for i in range(self.params['num_topics']):
			self.topic_name[i] = ldamodel.show_topic(i, topn=100)
		with open(self.params['topic_file'], 'w') as f:
			f.write(str(ldamodel.print_topics(num_topics=-1, num_words=10)))
		print('lda: finished.')

		counter = 0
		for doc in corpus:
			topics = ldamodel.get_document_topics(doc, minimum_probability=1e-4)
			topics.sort(key=lambda tup: tup[1], reverse=True)
			if len(topics) >= 1:
				self.topic_business[topics[0][0]].add(counter)
			counter += 1

		with open('models/step2.pkl', 'wb') as f:
			pickle.dump(self, f)
		print('step2: finished processing the reviews of %d businesses.' % counter)

	def step3(self):
		pass

if __name__ == '__main__':
	params = {}
	#public parameters
	params['content_file'] = 'models/content_file.txt'
	params['topic_file'] = 'models/topic_file.txt'
	params['num_topics'] = 100

	#dblp parameters
	params['dblp_files'] = ['../dblp-ref/dblp-ref-0.json', '../dblp-ref/dblp-ref-1.json', '../dblp-ref/dblp-ref-2.json', '../dblp-ref/dblp-ref-3.json']
	params['author_file'] = '../clus_dblp/vocab-'
	params['label_type'] = 'label'

	#yelp parameters
	params['yelp_business'] = '../yelp_data/business.json'
	params['yelp_user'] = '../yelp_data/user.json'
	params['yelp_checkin'] = '../yelp_data/checkin.json'
	params['yelp_review'] = '../yelp_data/review.json'
	params['content_file'] = 'models/content_file.txt'


	if not os.path.exists('models/step1.pkl'):
		cube = YelpCube(params)
		cube.step1()
	elif not os.path.exists('models/step2.pkl'):
		with open('models/step1.pkl', 'rb') as f:
			cube = pickle.load(f)
		cube.step2()
	elif not os.path.exists('models/step3.pkl'):
		with open('models/step2.pkl', 'rb') as f:
			cube = pickle.load(f)
		cube.step3()
	else:
		print('all 3 steps have finished.')


