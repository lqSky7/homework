i = str(input())
if len(i) != 10:
    print("Invalid length")
    exit()

for l in range(0,4):
    if not(ord(i[l]) >= 65 or ord(i[l]) <= 90):
        print("Invalid1")
        exit()

for f in range(5,9):
    if not(ord(i[f]) >= 48 or ord(i[f]) <= 57):
        print("Invalid2")
        exit()
    elif not(ord(i[-1]) >= 65 or ord(i[-1]) >= 90):
        print("Invalid3")
        exit()
    else:
        # print(ord(i[l]))
        # print(ord(i[-1]))
        print("Valid")
        break

