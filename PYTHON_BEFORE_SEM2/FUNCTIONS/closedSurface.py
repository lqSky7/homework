# dgits: consider closed surfaces.
def closedsr(num) -> int:
    if(num == 1 or num ==  2 or num == 3 or num == 5 or num == 7):
        return 0
    elif(num == 4 or num == 6 or num == 9 or num == 0):
        return 1
    elif(num == 8):
        return 2
    else:
        return -1

i = int(input("enter num"))
if(len(str(i)) != 1):
    print("invalid")
    exit()
print(closedsr(i))
