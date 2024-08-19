p = "rfscd/evfjk/qacj"
# print(p)
# print(len(p))

g = "rfscd//evfjk//qacj"
# print(g)
# print(len(g))

g = p*4

# print (not("rfscd" in g))
cnt=0
for e in g:
    if e == "/":
        cnt+=1

# print(cnt)

sliced_g = g[2:10]
# print(sliced_g)

char = "f"
chrord = ord(char)+1
# print(chr(chrord))

g = p+g
g = g.replace("/", "  ")
# print(g)
g = g.upper()
# print(g)
g = g.strip()   
# print(g)

print(g.endswith("ACJ"))
print(g.isupper())
