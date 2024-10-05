import re

i = input()

if(re.match("\A[D]([0-2]\d|[3][01])([0]\d|[1][012])-[AEIOU]\Z", i)):
    print("VALID")
else:  
    print("INVALID")