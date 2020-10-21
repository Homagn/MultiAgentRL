import random
import numpy as np

a=random.sample((1,2,3),2)
b=random.sample((1,2,3),3)
c=list(np.setdiff1d(b,a))
d=np.where(b==c[0])
e=[int(d[0][0])+1]
f=np.setdiff1d(b,e)-1
if(e[0]==2):
	f=f[0]
if(e[0]==1):
	f=f[1]
try:
	g=random.sample(tuple(f),1)[0]
except:
	g=f
b.insert(g,c[0])
color_assign = a+b
