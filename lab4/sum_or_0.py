sum = 0
average = 0

while True:
    k = int(input())
    if k == 0:
        average = sum / 2       
        print(sum, average)
        break;
    else:
        sum += k
        continue