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

		self.business = []
		self.user = []
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
		business_all = set()
		business_ca_takeout = set()
		business_il_goodforkids = set()
		num_business = 0
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

				business_all.add(p['business_id'])
				if p['state'].lower() == 'ca' and 'RestaurantsTakeOut' in p['attributes'] and p['attributes']['RestaurantsTakeOut']:
					business_ca_takeout.add(p['business_id'])
				if p['state'].lower() == 'il' and 'GoodForKids' in p['attributes'] and p['attributes']['GoodForKids']:
					business_il_goodforkids.add(p['business_id'])

				for cat in map(lambda x: x.lower(), p['categories']):
					if cat not in self.category_name:
						self.category_name.append(cat)
						self.category_business.append(set())
					self.category_business[self.category_name.index(cat)].add(p['business_id'])
				if p['city'] not in self.city_name:
					self.city_name.append(p['city'])
					self.city_business.append(set())
				self.city_business[self.city_name.index(p['city'])].add(p['business_id'])

		self.business = list(business_all)
		self.business_user = [set() for i in range(len(self.business))]
		del business_all
		print('finised input businesses: %d/%d' % (len(self.business), num_business) )

		#input users
		user_all = set()
		num_user = 0
		with open(self.params['yelp_user'], 'r') as f:
			for line in f:
				p = json.loads(line)
				num_user += 1
				if ('user_id' not in p) \
				or ('review_count' not in p) \
				or (int(p['review_count']) < 10):
					continue

				user_all.add(p['user_id'])

		self.user = list(user_all)
		self.user_business = [set() for i in range(len(self.user))]
		del user_all
		print('finised input users: %d/%d' % (len(self.user), num_user) )


		#input reviews
		num_review = 0
		with open(self.params['yelp_review'], 'r') as f, open(self.params['content_file'], 'w') as cf:
			for line in f:
				p = json.loads(line)
				num_review += 1
				if num_review % 1000 == 0:
					print('%d reviews processed' % num_review)
				if ('business_id' not in p) \
				or ('user_id' not in p) \
				or ('text' not in p) \
				or (not re.match("^[\w\s,.:?-]+$", p['text'])):
					continue
			
				try:
					bid = self.business.index(p['business_id'])
					uid = self.user.index(p['user_id'])
				except ValueError:
					continue

				cf.write(p['text']+'\n')
				self.review_business.append(p['business_id'])
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
			basenet['set0_net0'] = set()
			basenet['set0_net1'] = set()
			basenet['set1_net0'] = set()
			basenet['set1_net1'] = set()
			basenet['set0_test'] = set()
			basenet['set1_test'] = set()

			for b in map(lambda x: self.business.index(x), self.business_ca_takeout):
				if random.random() < 0.5:
					basenet['set0_net0'].add(b)
					for u in self.business_user[b]:
						if random.random() < 0.2:
							basenet['set0_test'].add((b, u))

			starttime = time.time()
			for b in basenet['set0_net0']:
				for u in self.business_user[b]:
					for b1 in self.user_business[u]:
						basenet['set0_net1'].add(b1)

			print('generated basenet 0 with net0/net1/test as %d/%d/%d' %(len(basenet['set0_net0']), len(basenet['set0_net1']), len(basenet['set0_test'])))
			print('time spent %f s' % (time.time()-starttime))

			for b in map(lambda x: self.business.index(x), self.business_il_goodforkids):
				if random.random() < 0.5:
					basenet['set1_net0'].add(b)
					for u in self.business_user[b]:
						if random.random() < 0.2:
							basenet['set1_test'].add((b, u))

			starttime = time.time()
			for b in basenet['set1_net0']:
				for u in self.business_user[b]:
					for b1 in self.user_business[u]:
						basenet['set1_net1'].add(b1)
			print('generated basenet 1 with net0/net1/test as %d/%d/%d' %(len(basenet['set1_net0']), len(basenet['set1_net1']), len(basenet['set1_test'])))
			print('time spent %f s' % (time.time()-starttime))
			
			with open('models/basenet.pkl', 'wb') as f:
				pickle.dump(basenet, f)

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


