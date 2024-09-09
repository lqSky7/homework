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
    mapd[str(i+1)] = letter

for k in range(18):
    letter = chr(ord('i')+k)
    mapd[f"{k+9}#"] = letter
print(mapd)




def decode(str, map):
    i = 0
    result = ""
    while(i< len(str)):
        if(i + 2 < len(str)) and str[i+2] == "#":
          
            reqStr = str[i:i+3]
            i += 3
            if reqStr in map.keys():
                result += map[reqStr]
                continue
        else: #then value is single dgt
            reqStr = str[i]
            i += 1
            if reqStr in map.keys():
                result += map[reqStr]
                continue

    return result
                    
print(decode(a, mapd))