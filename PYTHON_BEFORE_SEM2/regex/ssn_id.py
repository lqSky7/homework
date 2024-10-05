import re

b = str(input())

if(re.match(r"\A[A-Za-z]{4}[0-9]{4}", b)):
    if(b[0:4] == "TEMP"):
        print("Temp Resident, VALID")
    else:
        print("VALID")
else:
    print("INVALID")
    