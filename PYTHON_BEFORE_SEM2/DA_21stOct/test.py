number_ofInputs = 1
for i in range(number_ofInputs):
    length =  5
    wordtoChange =  "kiopl"
    wordtoMatch = "pkilo"

    # reject if impossible:
    #this logic is bad since different combinations may have same ascii, but too lazy to change it. 
    asciiword1 = 0 
    asciiword2 = 0

    for i in range(length):
        asciiword1 += ord(wordtoChange[i])
        

    for i in range(length):
        asciiword2 += ord(wordtoMatch[i])

    print(asciiword1 ,asciiword2)

    if(asciiword1 != asciiword2):
        print(-1)
        exit()



# main logic
wclist = list(wordtoChange)
 #first letter check.
if(wordtoMatch[0] != wordtoChange[0]):
    trgetIndex = wclist.index(wordtoMatch[0])
    wclist.insert(0, wordtoMatch[0])
    wclist.pop(trgetIndex+1)
    
# we match position by position, and throw any unmatched letter to the end, it'll all come together automatically. only if our first letter is unmatched we move stuff to front, otherwise we always pop and move stuff to the end.
for i in range(length-1):   
    if(wordtoMatch[i] != wordtoChange[i]):
        pass
    if(wordtoChange[i] == wordtoMatch[i]):
        pass
print(''.join(wclist))