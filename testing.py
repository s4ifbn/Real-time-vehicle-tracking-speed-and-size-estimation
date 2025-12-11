import math
from sklearn import datasets    # ML library

data = datasets.load_iris()
labels = data["target"]
features = data["feature_names"]
print(data["target_names"])
print(labels)
print(features)

r = ((80*8) + (91*8) +(81*6)+(82*4) + (78*2) + (70*2))/30

a = (125 % 26)
b = (5 ** 6)

#print((42 ** 9) % 53)
#print(math.sqrt(26 + 169))

# u =1.3063778838630806904686144926   # Mills Number
# print((math.floor(u**81)))