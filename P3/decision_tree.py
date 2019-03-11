import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			C=len(branches)
			B=len(branches[0])
			branches=np.asarray(branches)
			sum_all=branches.sum()
			if (sum_all==0):
				return 0
			entropy=0.0
			for i in range (0,B):
				probability=branches[:,i]/sum(branches[:,i])
				for j in range(0,len(probability)):
					if (probability[j]!=0):
						probability[j]=branches[j,i]*np.log2(probability[j])
					else:
						probability[j]=0
				entropy -= sum(probability)
			entropy=entropy/sum_all
			return entropy

		features=np.asarray(self.features)
		number_n,number_features=features.shape
		if (number_features==0):
			return self.cls_max
		best_entropy=[]
		for idx_dim in range(len(self.features[0])):
			X0=features[:,idx_dim]
			B=np.unique(X0)  #number of features
			C=np.unique(self.labels) #number of class
			branches=np.zeros((len(C),len(B)),dtype=int)
			dic={}
			dic1={}
			for i in range(0,len(B)):
				dic[B[i]]=i      # create a list of feature
			for i in range(0,len(C)):
				dic1[C[i]]=i     # create a list of class
			for i in range(0,len(self.labels)):
				x_index=dic1[self.labels[i]]     #to get class label
				y_index=dic[X0[i]]               # to get figure label
				branches[x_index][y_index]+=1
			entropy=conditional_entropy(branches.tolist())	
			best_entropy.append(entropy)
		
		self.dim_split=np.argmin(best_entropy)

		############################################################
		# TODO: split the node, add child nodes
		############################################################
		B=np.unique(features[:,self.dim_split])
		X1=features[:,self.dim_split]
		self.feature_uniq_split=np.unique(X1).tolist()
		np.delete(features,self.dim_split,1).tolist()
		if (len(B)<2):
			self.splittable = False
		else:
			for i in B:
				self.children.append(TreeNode(features[np.where(X1==i)].tolist(),np.array(self.labels)[np.where(X1==i)].tolist(),self.num_cls))
					



		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



