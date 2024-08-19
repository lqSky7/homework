import math

a = float(input("enter a: "))
b = float(input("enter b: "))
c = float(input("enter c: "))

root1 = (-b + math.sqrt(b**2 - 4*a*c)) / (2*a)
root2 = (-b - math.sqrt(b**2 - 4*a*c)) / (2*a)

print("Root 1 =", root1)
print("Root 2 =", root2)
