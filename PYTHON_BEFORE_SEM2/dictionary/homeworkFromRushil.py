# inputStr = int(input())
# we read string from opposite side to easily handle # stuff
a = "1326#11#4"
b = "12#57"
c = "013#"
d = "1234657#"

def edgeCase(str):
    for i in range(len(str)-1,-1,-1):
        if str[i]=="0":
            return False
        
    return True
    
def HashValidity(str):
    for i in range(len(str)-1, 1,-1):
        if(str[i]=="#"):
            if(int(str[i-1])>=0 and int(str[i-1])<=6 and int(str[i-2])>=1 and int(str[i-2])<=2):
                return True
            else:
                return False
        
        return True

valuePairs = {}
for i in range(9):
    letter = chr(ord('a')+ i)
    valuePairs[letter] = str(i+1)

for i in range(16):
    letter = chr(ord('i')+ i)
    valuePairs[letter] = f"{i+10}#"


# #check 
# if(edgeCase(a) or not HashValidity(a) == False):
#     print("invalid")
#     exit()



# //break stuff to 2 before 
def decode(string, map):
    i = 0
    result = ""
    
    while i < len(string):
        
        if i + 2 < len(string) and string[i + 2] == "#":
            testStr = string[i:i+3]  
            if testStr in map.values():
                for z, j in map.items():
                    if j == testStr:
                        result += z  
                        
            i += 3  
        else:
            testStr = string[i]  
            if testStr in map.values():
                for z, j in map.items():
                    if j == testStr:
                        result += z  
                        
            i += 1  

    return result




print(decode(a, valuePairs))
    
