exampleTupple = (1, 2, 3, 4, 5, 4, 7, 8, 9, 9)
for i in range(0, len(exampleTupple)):

    
    if exampleTupple.count(exampleTupple[i]) > 1:
        print(exampleTupple[i])