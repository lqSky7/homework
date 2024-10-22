number_ofInputs = int(input())
for i in range(number_ofInputs):
    length =  int(input())
    wordtoChange =  input()
    wordtoMatch = input()

    # reject if impossible:
    asciiword1 = 0 
    asciiword2 = 0

    for i in range(length):
        asciiword1 += ord(wordtoChange[i])
        

    for i in range(length):
        asciiword2 += ord(wordtoMatch[i])

    if(asciiword1 != asciiword2):
        print(-1)
        exit()

#print(asciiword1 ,asciiword2)

# main logic

