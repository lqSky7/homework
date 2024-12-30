# sorting dictionary by value:
# 

exampl = {4: 3, 5:6, 7:3, 7:9}
print(exampl)

max = 0
valueList = []
keyList = []

for i,j in exampl.items():
    valueList.append(j)
    keyList.append(i)

valueList.sort(reverse=True)

returndct = {}

for i in range(len(keyList)):
    returndct.update({keyList[i]: valueList[i]})