import numpy as np
import pandas as pd

f = open("formula2.csv", "w")
f.write("w.x.y.z\n")

np.random.seed(40)

w = ["young"]*50
x = 4*np.random.rand(50) -2
y = 2000*np.random.rand(50) - 1000
z = x + y + np.random.randn(50)

for v1,v2,v3,v4 in zip(w,x,y,z):
    f.write(v1 + "," + str(v2) + "," + str(v3) + "," + str(v4) + "\n")

w = ["old"]*50
x = 4*np.random.rand(50) -2
y = 2000*np.random.rand(50) - 1000
z = 3*x + 5*y + np.random.rand(50)


for v1,v2,v3,v4 in zip(w,x,y,z):
    f.write(v1 + "," + str(v2) + "," + str(v3) + "," + str(v4) + "\n")

f.close()

data = pd.read_csv("formula2.csv")
print(data)
