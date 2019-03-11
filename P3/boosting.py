import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
                Inputs:
                - features: the features of all test examples
   
                Returns:
                - the prediction (-1 or +1) for each example (in a list)
                '''
                ########################################################
                # TODO: implement "predict"
		prediction=[0.0]*len(features)
		for t in range(0,self.T):
			tmp=self.clfs_picked[t].predict(features)
			for j in range(0,len(features)):
				prediction[j] += self.betas[t]*tmp[j]
		for i in range(0,len(features)):
			if (prediction[i] <=0):
				prediction[i] = -1
			else:
				prediction[i] = 1
		return prediction
                ########################################################


	

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		n=len(labels)
		D=np.ones(n)/n
		et=None
		ht=None
		clfs_list=list(self.clfs)
		for iterations in range(0,self.T):
			errors=[]
			for j in range(0,self.num_clf):
				yn = clfs_list[j].predict(features)
				y_indicator=np.not_equal(labels,yn)
				errors.append(np.dot(D,y_indicator))
			et = min(errors)
			ht = errors.index(et)
			
			self.clfs_picked.append(clfs_list[ht])
			tmp=(1.0-et)/et
			bt=0.5*np.log(tmp)
			self.betas.append(bt)
		
			predictions=clfs_list[ht].predict(features)
			for k in range(n):
				if (predictions[k]==labels[k]):
					D[k]*=np.exp(-bt)
				else:
					D[k]*=np.exp(bt)
			
			D=D/(sum(D))
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	
