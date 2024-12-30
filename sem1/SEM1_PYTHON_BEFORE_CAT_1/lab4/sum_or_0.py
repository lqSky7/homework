sum = 0
average = 0
iterator = 0
while True:
    k = int(input())
    if k == 0:
        average = sum / iterator      
        print(sum, average)
        break;
    else:
        iterator += 1
        sum += k
        continue