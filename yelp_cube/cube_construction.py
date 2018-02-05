import pickle
import sys
import os
import re
import json
import random
from time import gmtime, strftime
from collections import defaultdict
from gensim import corpora, models
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
				if ('business_id' not in p) \
				or (p['business_id'] not in self.business) \
				or ('user_id' not in p) \
				or (p['user_id'] not in self.user) \
				or ('text' not in p) \
				or (not re.match("^[\w\s,.:?-]+$", p['text'])):
					continue

				cf.write(p['text']+'\n')
				self.review_business.append(p['business_id'])
				self.business_user[self.business.index(p['business_id'])].add(self.user.index(p['user_id']))
				self.user_business[self.user.index(p['user_id'])].add(self.business.index(p['business_id']))

		print('finised input reviews: %d/%d' % (len(self.review_business), num_review) )

		#input checkins
		num_checkin = 0
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
				#self.business_user[self.business.index(p['business_id'])].add(self.user.index(p['user_id']))
				#self.user_business[self.user.index(p['user_id'])].add(self.business.index(p['business_id']))

		print('finised input reviews: %d/%d' % (valid_checkin, num_checkin) )

		with open('models/step1.pkl', 'wb') as f:
			pickle.dump(self, f)

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
			for b in basenet['set0_net0']:
				for u in self.business_user[b]:
					for b1 in self.user_business[u]:
						basenet['set0_net1'].add(b1)
			print('generated basenet 0 with net0/net1/test as %d/%d/%d' %(len(basenet['set0_net0']), len(basenet['set0_net1']), len(basenet['set0_test'])))

			for b in map(lambda x: self.business.index(x), self.business_il_goodforkids):
				if random.random() < 0.5:
					basenet['set1_net0'].add(b)
					for u in self.business_user[b]:
						if random.random() < 0.2:
							basenet['set1_test'].add((b, u))
			for b in basenet['set1_net0']:
				for u in self.business_user[b]:
					for b1 in self.user_business[u]:
						basenet['set1_net1'].add(b1)
			print('generated basenet 1 with net0/net1/test as %d/%d/%d' %(len(basenet['set1_net0']), len(basenet['set1_net1']), len(basenet['set1_test'])))

			with open('models/basenet.pkl', 'wb') as f:
				pickle.dump(basenet, f)

		print('step1: finished.')

	def step2(self):
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
				if line_num % 2 == 1:
					texts.append(content)
					content = []
				line_num += 1
				#if line_num % 20000 == 0:
				#		print("step2: "+strftime("%Y-%m-%d %H:%M:%S", gmtime())+': processing paper '+str(line_num//2))

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
		for paper in corpus:
			topics = ldamodel.get_document_topics(paper, minimum_probability=1e-4)
			topics.sort(key=lambda tup: tup[1], reverse=True)
			if len(topics) >= 1:
				self.topic_author[topics[0][0]] |= self.paper_author[counter]
				for a1 in self.paper_author[counter]:
						for a2 in self.paper_author[counter]:
							if a1 != a2:
								self.topic_link[topics[0][0]][a1+','+a2] += 1
			#if len(topics) >= 2:
				#self.cell_content_two[topics[1][0]].append(counter)
			#if len(topics) >= 3:
				#self.cell_content_three[topics[2][0]].append(counter)
			counter += 1

		with open('models/step2.pkl', 'wb') as f:
			pickle.dump(self, f)
		print('step2: finished processing '+str(counter)+' papers.')

	def step3(self):
		print('step3: writing network files.')
		num_node = 0
		num_edge = 0
		with open('models/year_name.txt', 'w') as namef, open('models/year_node.txt', 'w') as nodef, open('models/year_link.txt', 'w') as linkf:
			for year in self.year_name:
				namef.write(str(year)+'\n')
				nodef.write(str(self.year_name.index(year))+'\n')
				num_node += 1
				for year_c in self.year_name:
					if abs(int(year) - int(year_c)) == 1:
						linkf.write(str(self.year_name.index(year))+'\t'+str(self.year_name.index(year_c))+'\t1\n')
						num_edge += 1
		print('step3: finished year network files with '+str(num_node)+' nodes and '+str(num_edge)+' edges.')

		num_node = 0
		num_edge = 0
		with open('models/venue_name.txt', 'w') as namef, open('models/venue_node.txt', 'w') as nodef, open('models/venue_link.txt', 'w') as linkf:
			for venue in self.venue_name:
				namef.write(''.join(venue.split())+'\n')
				nodef.write(str(self.venue_name.index(venue))+'\n')
				num_node += 1
				for venue_c in self.venue_name:
					if venue != venue_c:
						same = len(set(venue.split()) & set(venue_c.split()))
						if same > 0:
							linkf.write(str(self.venue_name.index(venue))+'\t'+str(self.venue_name.index(venue_c))+'\t'+str(same)+'\n')
							num_edge += 1
		print('step3: finished venue network files with '+str(num_node)+' nodes and '+str(num_edge)+' edges.')

		num_node = 0
		num_edge = 0
		with open('models/topic_name.txt', 'w') as namef, open('models/topic_node.txt', 'w') as nodef, open('models/topic_link.txt', 'w') as linkf:
			for ind in range(len(self.topic_name)):
				namef.write(str(self.topic_name[ind])+'\n')
				nodef.write(str(ind)+'\n')
				num_node += 1
				for ind_c in range(len(self.topic_name)):
					if ind != ind_c:
						words = map(lambda x: x[0], self.topic_name[ind])
						words_c = map(lambda x: x[0], self.topic_name[ind_c])
						same = len(set(words) & set(words_c))
						if same > 0:
							linkf.write(str(ind)+'\t'+str(ind_c)+'\t'+str(same)+'\n')
							num_edge += 1
		print('step3: finished topic network files with '+str(num_node)+' nodes and '+str(num_edge)+' edges.')

		self.venue_name
		self.year_name
		self.topic_name
		with open('models/step3.pkl', 'wb') as f:
			pickle.dump(self, f)

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


