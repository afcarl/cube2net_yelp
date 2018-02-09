import pickle
import time
from collections import defaultdict
from cube_construction import YelpCube
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
import numpy as np
import os
import implicit
from nmf_mask import mnmf

class YelpEval(object):
	#label_type:
		#group: small set of 116 authors
		#label: large set of 4236 authors
	def __init__(self, cube=None, business=[], setn=0):

		if cube == None:
			with open('models/step3.pkl', 'rb') as f:
				self.cube = pickle.load(f)
		else:
			self.cube = cube

		with open('models/basenet.pkl', 'rb') as f:
			basenet = pickle.load(f)

		if setn == 0:
			self.x = basenet['set0_business']
			self.y = basenet['set0_user']
			self.z = basenet['set0_link']
		else:
			self.x = basenet['set1_business']
			self.y = basenet['set1_user']
			self.z = basenet['set1_link']		

	
	def nodeGen(self, size=0):

		self.business = self.x.copy()
		self.user = self.y.copy()
		if size == 0:
			return

		for b in self.business:
			self.user |= self.cube.business_user[b]
		if size == 1:
			return

		for u in self.user:
			self.business |= self.cube.user_business[u]
		if size == 2:
			return

		for b in self.business:
			self.user |= self.cube.business_user[b]
		if size == 3:
			return

		for u in self.user:
			self.business |= self.cube.user_business[u]
		if size == 4:
			return

		for b in self.business:
			self.user |= self.cube.business_user[b]


	def netGen(self, size=0):

		self.nodeGen(size)
		self.b_id = list(self.business)
		self.u_id = list(self.user)		
		row = []
		col = []
		self.link = set()
		for i in range(len(self.b_id)):
			for j in range(len(self.u_id)):
				if self.u_id[j] in self.cube.business_user[self.b_id[i]] and (self.b_id[i], self.u_id[j]) not in self.z:
					row.append(i)
					col.append(j)
					self.link.add((self.b_id[i], self.u_id[j]))

		self.mat = coo_matrix((np.ones(len(row)), (np.array(row), np.array(col))), shape=(len(self.b_id), len(self.u_id)))

		print('Size of generated network: %d/%d/%d' % (len(self.business), len(self.user), len(row)))
	

	def netPred(self, method='mf', dim=100, alpha=0.1):
		'''
			supported methods: mf, cf, mnmf, fancy_nnm, fancy_soft
		'''
		if method == 'mf':
			model = NMF(n_components=dim, alpha=alpha, l1_ratio=0.2)
			W = model.fit_transform(self.mat)
			H = model.components_
			self.pred = np.matmul(W, H)
		elif method == 'cf':
			model = implicit.als.AlternatingLeastSquares(factors=dim, regularization=alpha)
			model.fit(self.mat)
			self.pred = np.matmul(model.item_factors, model.user_factors.T)
		elif method == 'mnmf':
			self.pred = mnmf(self.mat, dim, alpha)
		elif 'fancy' in method:
			X = self.mat.toarray().astype(np.float)
			X[X==0] = np.nan
			if 'nnm' in method:
				self.pred = NuclearNormMinimization(error_tolerance=0.01).complete(X)
			elif 'soft' in method:
				self.pred = SoftImpute().complete(X)

	def netEval(self, k=2):
		correct = 0
		for u in self.y:
			scores = self.pred[:, self.u_id.index(u)]
			preds = map(lambda x: self.b_id[x], list(scores.argsort()[::-1]))
			kk = 0
			i = 0
			while kk < k:
				if (preds[i], u) not in self.link and preds[i] in self.x:
					kk += 1
					if (preds[i], u) in self.z:
						correct += 1
				i += 1

		prec = correct*1.0/(k*len(self.y))
		rec = correct*1.0/len(self.z)

		ix = np.unravel_index(self.pred.argsort(axis=None), dims=self.pred.shape)
		ids = zip(*ix)[::-1]
		tp = []
		fp = []
		ctp = 0
		cfp = 0
		for t in ids:
			if (self.b_id[t[0]], self.u_id[t[1]]) not in self.link and self.b_id[t[0]] in self.x and self.u_id[t[1]] in self.y:
				if (self.b_id[t[0]], self.u_id[t[1]]) in self.z:
					ctp += 1
				else:
					cfp += 1
				tp.append(ctp)
				fp.append(cfp)
		tp = map(lambda x: x*1.0/ctp, tp)
		fp = map(lambda x: x*1.0/cfp, fp)
		auc = 0
		for i in range(1, len(tp)):
			auc += (fp[i]-fp[i-1])*tp[i]

		print('Evaluation results: %f/%f/%f' %(prec, rec, auc))
		return((prec, rec, auc))

	def netDebug(self):
		for t in self.link:
			print(self.pred[self.b_id.index(t[0]), self.u_id.index(t[1])])
		print(self.pred)

	def noCubeEval(self, size=0, method='mf', dim=100, alpha=0.1, k=10):
		
		starttime = time.time()
		self.netGen(size)
		print('network generation time: %ds' % (time.time()-starttime))
		starttime = time.time()
		self.netPred(method, dim, alpha)
		print('prediction time: %ds' % (time.time()-starttime))
		self.netEval(k)
		#self.netDebug()

if __name__ == '__main__':
	test = YelpEval()
	test.noCubeEval(size=5, method='fancy_soft', dim=100, alpha=0.1, k=5)


