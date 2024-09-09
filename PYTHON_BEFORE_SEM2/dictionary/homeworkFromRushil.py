# inputStr = int(input())
# we read string from opposite side to easily handle # stuff
a = "1326#11#4"
b = "12#57"
c = "013#"
d = "1234657#"


# //first create the map

mapd = {}
for i in range(9):
    letter = chr(ord('a')+ i)
    mapd[letter] = str(i+1)

for k in range(18):
    letter = chr(ord('i')+k)
    mapd[letter] = f"{k+9}#"
print(mapd)




def decode(str, map):
    i = 0
    result = ""
    while(i< len(str)):
        if(i + 2 < len(str)) and str[i+2] == "#":
          
            reqStr = str[i:i+3]
            i += 3
            if reqStr in map.values():
                for k,l in map.items():
                    if(l == reqStr):
                        result += k
                        continue
        else: #then value is single dgt
            reqStr = str[i]
            i += 1
            if reqStr in map.values():
                for k,l in map.items():
                    if(l == reqStr):
                        result += k
                        continue

    return result
                    
print(decode(a, mapd))