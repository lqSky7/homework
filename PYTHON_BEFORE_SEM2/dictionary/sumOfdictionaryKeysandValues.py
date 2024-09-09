
exampl = {4: 3, 5:6, 7:3, 7:9}
sumkey = 0
sumVal = 0
for i,j in exampl.items():
    print("Key :",i, "Value: ",j)
    sumkey += i
    sumVal += j

print(sumkey, sumVal)