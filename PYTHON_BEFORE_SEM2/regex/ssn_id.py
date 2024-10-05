import re

b = str(input())

if len(b) != 8 or "@#$%^&*()!~" in b:
    print("Invalid SSN")
    exit()


if(re.match(r"\A[A-Za-z]{4}[0-9]{4}", b)):
    if(b[0:4] == "TEMP"):
        print("Temp Resident, VALID")
    else:
        print("VALID")
else:
    print("INVALID")
