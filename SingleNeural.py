from matplotlib import pyplot as plt
import numpy as np

def nonline (x,deriv=False):
	if(deriv == True):
		return x*(1-x)

	return 1/(1+np.exp(-x))

X=np.array([[0,0,1],
			[0,1,1],
			[1,0,1],
			[1,1,1]
			])

Y =np.array([[0],
			 [1],
			 [1],
			 [0]])

np.random.seed(1)

syn0=2 * np.random.random((3, 4)) - 1
syn1=2 * np.random.random((4, 1)) - 1

epoch = 600000
for j in range(epoch):
	l0=X
	l1 = nonline(np.dot(l0, syn0))
	l2 = nonline(np.dot(l1, syn1))

	l2_error = (Y - l2)

	if(j%10000) == 0:
		print ("Error:",str(np.mean(np.abs(l2_error))))

	l2_delta = l2_error * nonline(l2,deriv=True)
	
	l1_error = l2_delta.dot(syn1.T)

	l1_delta = l1_error * nonline(l1,deriv=True)

	#Update
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

print ("Ouput after Training")
print (l2)