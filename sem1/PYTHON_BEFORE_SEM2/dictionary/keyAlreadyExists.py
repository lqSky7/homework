# key already exists
# 
hi = int(input())
def genmap(N):

    for i in range(hi+1):
        N[i] = i*i


exampl = {}
genmap(exampl)

keyList = []
for i,j in exampl.items():
    keyList.append(i)

keyList.sort()
for i in range(len(keyList)):
    if(i == len(keyList)-1):
        break

    if (keyList[i]==keyList[i+1]):
        print("repeat")
        exit
print("No repetition")
