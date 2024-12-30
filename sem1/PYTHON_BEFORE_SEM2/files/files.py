with open('example.txt', 'w') as file:
    file.write('Hello, world!\n')
    file.write('This is a test file.\n')
    file.write('Each line will be read and processed.\n')

with open('example.txt', 'r') as file:
    print(file.read())

with open('example.txt', 'r') as file:
    print(file.readline())

with open('example.txt', 'r') as file:
    print(file.readlines())

with open('example.txt', 'r') as file:
    print(file.tell())
    file.seek(7)
    print(file.tell())
    print(file.read())



#HOMEWORK Qu
def facn(num):
    if num == 0:
        return 1
    else:
        # print(num)
        return num * facn((num-1))

hi = facn(5)
print(hi)

#lambda 1
x = lambda a: a + 15
k = x(5)
print(k)

# lambda 2
l = lambda a,b: a*b
n = l(3,7)
print(n)

# lambda 3
def doubler(ww):
    return lambda c: ww*c

i1 = doubler(2)
m = i1(5)
print(m)

f = open(r"C:\Users\ex1\Documents\Ashish_WebDev\marks.txt", "r")
z = f.readlines()

print(z)
cnt = 0
for i in z:
    i = "".join((i).split())
    if(i == "100" or i == 100):
        cnt+=1

print("\n",cnt)
