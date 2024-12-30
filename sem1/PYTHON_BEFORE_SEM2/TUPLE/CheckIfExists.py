exampletuple = (1, 2, 3, 4, 5)
i = input("Enter the element to check")
for k in range(0, len(exampletuple)):
    if (exampletuple[k] == i):
        print("Element exists")
        break