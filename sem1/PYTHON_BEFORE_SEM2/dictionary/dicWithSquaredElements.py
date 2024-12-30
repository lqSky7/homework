hi = int(input())
def genmap(N):

    for i in range(hi+1):
        N[i] = i*i


exampl = {}
genmap(exampl)
for i,j in exampl.items():
    print("Key :",i, "Value: ",j)
