import numpy as np
from sklearn.cluster import KMeans


'''
unsupervised model :  understand data from observation
'''

class Gaussian_Mixture_Model(object):
    def __init__(self,m,threshold=1e-4):
        self.m=m
        self.threshold=threshold #stop threshold

    def __call__(self,n,space_num=10000):
        ''':
        after training : get coef,mu,cov
        sampling : sampling n data based on gaussian mixture distribution
        gaussian mixture distribtuion:sum over all k --- coef_k*N(xi|u_k,cov_k)
        '''

        min=self.x.min(axis=0)
        max=self.x.max(axis=0)

        #random choose a point in a specific dimension
        space=np.array([np.random.choice(np.linspace(np.floor(min[d]), np.floor(max[d]), num=space_num),size=space_num)\
                        for d in range(self.x.shape[-1])]).T  #(space_num,dim)
        prob=np.zeros((space_num))

        for idx,xi in enumerate(space):
            xi=xi[np.newaxis,:] #(1,d)
            mixture_prob=np.sum(self.coef*self.multivariate_gaussian_pdf(xi,self.mu,self.cov))
            prob[idx]=mixture_prob

        prob=prob/prob.sum()
        sample_idx=np.random.choice(range(0,space_num),p=prob,size=n)

        sample=space[sample_idx,:]

        return sample


    def init_parameters(self,x):
        ''':parameter
        initialize (aimed to speed up convergence):
        coef: % of each cluster number
        mu:midpoint of each cluster
        covariance matrix sigma: covariance of each cluster
        '''
        try:
            assert len(x.shape)==2
        except:
            raise ValueError("x dimension can't not equal to 2")

        n, dim = x.shape

        kmeans=KMeans(n_clusters=self.m)
        pred=kmeans.fit_predict(x)

        #m mixture ceofficient
        count=np.array([(pred==k).sum() for k in range(self.m)])
        coef=count/count.sum() #(m,)

        #m parameters of gaussian distributions
        mu=kmeans.cluster_centers_ #(m,d)
        cov=np.array([np.cov(np.array([x[idx,:] for idx in np.where(pred==k)[0]]).T) for k in range(self.m)])

        return coef,mu,cov


    def multivariate_gaussian_pdf(self,x,mu,cov):
        ''':parameter

        intuition:xi在某一群的機率是多少,如果離mu越遠機率越小
        x:(1,d)
        mu:(m,d)
        cov:(m,d,d)
        '''
        x=x[:,np.newaxis,:] #(1,1,d)
        mu=mu[:,np.newaxis,:] #(m,1,d)
        dim=x.shape[-1]
        term1=np.power(2*np.pi,-0.5*dim)*np.power(np.linalg.det(cov),-0.5) #(m,)
        term2=np.exp(-0.5*np.matmul(np.matmul((x-mu),np.linalg.inv(cov)),np.transpose(x-mu,axes=(0,2,1)))) #(m,1,1)

        return term1*(term2.flatten()) #(m,)


    def E_step(self,x,coef,mu,cov):
        ''':parameter
        x:(n,d)
        coef:(m,)
        mu:(m,d)
        cov:(m,d,d)

        return post prob
        '''
        for idx,xi in enumerate(x):#xi:(d,)
            xi=xi[np.newaxis,:] #(1,d)
            prob=self.multivariate_gaussian_pdf(xi,mu,cov) #(m,)
            denominator=np.sum(coef*prob) #scaler
            molecular=coef*prob #(m,)
            wi=molecular/denominator  #(m,)
            if idx==0:
                w=wi[np.newaxis,:] #(1,m)
            else:
                w=np.vstack((w,wi))
        return w #(n,m)


    def M_step(self,x,w):
        ''':parameter
        x:(n,d)
        w:(n,m)
        (x-mu) (x-mu).T
        '''
        n=w.shape[0]
        sum_w=np.sum(w,axis=0) #(m,)
        coef=sum_w/n #(m,)


        mu=np.matmul(w.T,x)/sum_w[:,np.newaxis] #(m,d)


        cov=[] #(m,d,d)
        for k in range(self.m):
            cov_k=0
            sum_wi=0
            for i in range(x.shape[0]):
                coef_ik=w[i,k]
                sum_wi+=coef_ik
                residual=(x[i,:]-mu[k,:])[:,np.newaxis] #(d,1)
                cov_k+=coef_ik*np.matmul(residual,residual.T) #(d,d)
            cov.append(cov_k/sum_wi)
        cov=np.array(cov) #(m,d,d)

        return coef,mu,cov


    def fit_data_distribution(self,x):
        '''
        EM-Algorithm
        '''
        coef,mu,cov=self.init_parameters(x)

        cov_prev=cov.copy()
        i=0
        while True:
            w=self.E_step(x,coef,mu,cov) #expectation
            coef, mu, cov=self.M_step(x,w)  #maximization

            # convergence criterion
            if np.mean(np.abs(cov-cov_prev))<self.threshold:
                break
            else:
                cov_prev = cov.copy()
        self.coef=coef #(m,)
        self.mu=mu #(m,d)
        self.cov=cov  #(m,d,d)
        self.x=x #(n,d)






