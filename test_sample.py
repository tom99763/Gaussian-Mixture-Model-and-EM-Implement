import numpy as np
import matplotlib.pyplot as plt
from gmm import Gaussian_Mixture_Model



n_samples = 50
# generate random sample, two components
np.random.seed(0)
# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
# concatenate the two datasets into the final training set
x = np.vstack([shifted_gaussian, stretched_gaussian])

print('fit data distribution...')
gmm=Gaussian_Mixture_Model(m=2,threshold=1e-4)
gmm.fit_data_distribution(x)

print('sampling...')
sample=gmm(n=200,space_num=10000)

plt.scatter(sample[:,0],sample[:,1],color='red',edgecolor='black',marker='s',s=40,label='generated data')
plt.scatter(x[:,0],x[:,1],color='cyan',edgecolor='black',marker='s',s=40,label='real data')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('gmm+em generated example')
plt.show()

