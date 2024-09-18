n = str(input())
if(len(n) <=2 ):
    print("invalid")
    exit()
rev = ""
if(len(n) == 3):
    print(n)

for i in range(len(n)-2):
    rev+=n[len(n)-3-i]    
    
rev += n[-1]
    
rev += n[-2]
print(rev)
