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


#pseudocode
Define fn 'closedsr' with parameter 'num' returning integer:
    If 'num' is 1, 2, 3, 5, or 7:
        Return 0
    Else if 'num' is 4, 6, 9, or 0:
        Return 1
    Else if 'num' is 8:
        Return 2
    Otherwise:
        Return -1

input number and store it in 'i'
If the length of 'i' as a string is not 1:
    Print "invalid"
    exitt

Print the result of calling func 'closedsr'

