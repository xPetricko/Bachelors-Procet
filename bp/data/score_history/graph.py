import matplotlib.pyplot as plt
import random as r
NO = 7
NO1 = 6
NO2 = 3
# with open(str(NO)+"_score_history.txt", "r") as f:
#     a = f.readlines()

# b = [float(x) for x in a]

with open(str(NO2)+"_runnig_history.txt", "r") as f:
    a = f.readlines()

b = [float(x) for x in a]

with open(str(NO1)+"_runnig_history.txt", "r") as f:
    a = f.readlines()

d = [float(x) for x in a]

with open(str(NO)+"_runnig_history.txt", "r") as f:
    a = f.readlines()

c = [float(x) for x in a]

plt.plot(range(len(b)), b, label="M1 priemerná")
plt.plot(range(len(d)), d, label="M2 priemerná")
plt.plot(range(len(c[:3212])), c[:3212], label="M4 priemerná")
plt.ylabel("Epizóda")
plt.xlabel("Odmena")
plt.legend(loc="upper left")
plt.show()
