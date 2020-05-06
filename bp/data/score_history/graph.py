import matplotlib.pyplot as plt

NO = 8
with open(str(NO)+"_runnig_history.txt","r") as f:
    a = f.readlines()

b=[]

for i in a:
    b.append(float(i[:-1]))

plt.plot(b)
plt.show()


