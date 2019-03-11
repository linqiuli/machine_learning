import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            clf=KMeans(self.n_cluster,self.max_iter,self.e)
            self.means,rk,_=clf.fit(x)
            self.pi_k=np.zeros(self.n_cluster)
            self.variances=np.zeros((self.n_cluster,D,D))
            for i in range(self.n_cluster):
                N_k=np.where(rk==i)
                self.pi_k[i]=float(len(N_k[0])) / float(N)
                self.variances[i]=np.sum([np.outer(x[j]-self.means[i],x[j]-self.means[i]) for j in N_k[0]],axis=0)
                self.variances[i]=self.variances[i]/float(len(N_k[0]))
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means=np.random.rand(self.n_cluster,D) 
            self.variances=np.zeros((self.n_cluster,D,D))
            for i in range(0,self.n_cluster):
                self.variances[i]=np.identity(D)
            self.pi_k=np.ones(self.n_cluster)
            self.pi_k=1.0/float(self.n_cluster)*self.pi_k
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        l_old=self.compute_log_likelihood(x,self.means,self.variances,self.pi_k)
        yik=np.zeros((N,self.n_cluster))
        for i in range(0,self.max_iter):
            ###E-step 
            for k in range(self.n_cluster):
                sig = self.variances[k]
                mu = self.means[k] 
                while np.linalg.matrix_rank(sig) < D:
                    sig += 0.001*np.identity(D)
                sig_inv = np.linalg.inv(sig)
                sig_det = np.linalg.det(sig)
                mu = np.matmul(np.ones((N,1)),mu.reshape(1,D))
                dxu = x - mu
                ind = -0.5*np.sum( np.matmul(dxu, sig_inv) * dxu, 1)
                yik[:,k] = self.pi_k[k]*np.exp(ind)/(np.sqrt((2*np.pi)**D*sig_det))

            yik = yik/np.sum(yik,1)[:,None] ## normalize each row
            N_k = np.sum(yik,0)
            self.pi_k = N_k/N ## mixture weight

            self.means = np.matmul(yik.T, x)/N_k[:,None] ## means

            for k in range(self.n_cluster):# compute self.variances:
                mu = self.means[k]
                self.variances[k] = np.sum([yik[i][k]*np.outer(x[i]-mu, x[i]-mu) for i in range(N)], 0)/N_k[k]

            l=self.compute_log_likelihood(x,self.means,self.variances,self.pi_k)
            if (abs(l-l_old)<=self.e):
                break
            else:
                l_old=l
        return i+1
        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        yk=np.random.choice(self.n_cluster,N,p=self.pi_k)
        _,D=self.means.shape
        samples=np.zeros((N,D))
        for n in range(N):
            k=int(yk[n])
            samples[n]=np.random.multivariate_normal(self.means[k],self.variances[k])
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        N,D=x.shape
        yik=np.zeros((N,self.n_cluster))
        for k in range(self.n_cluster):
            sig = variances[k]
            mu = means[k]
            while np.linalg.matrix_rank(sig) < D:
                sig += 0.001*np.identity(D)
            sig_inv = np.linalg.inv(sig)
            sig_det = np.linalg.det(sig)
            mu = np.matmul(np.ones((N,1)),mu.reshape(1,D))
            dxu = x - mu
            ind = -0.5*np.sum( np.matmul(dxu, sig_inv) * dxu, 1)
            yik[:,k] = pi_k[k]*np.exp(ind)/(np.sqrt((2*np.pi)**D*sig_det))

        log_likelihood = float(np.sum(np.log(np.sum(yik,1))))
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            D,_=self.variance.shape
            while(np.linalg.det(self.variance)==0):
                self.variance+=10**(-3)*np.identity(D)
            self.inv=np.linalg.inv(self.variance)
            self.c=np.linalg.det(self.variance)*(2*np.pi)**(float(D)/2.0)
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            tmp=-0.5*np.dot(np.dot(x-self.mean,self.inv),(x-self.mean))
            p=np.divide(np.exp(tmp),self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
