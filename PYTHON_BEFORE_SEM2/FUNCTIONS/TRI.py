import math

def linecheck(a, b, c) -> bool:
    if a[0] == b[0] == c[0]:
        return False
    if a[1] == b[1] == c[1]:
        return False
    return True

def distanceFormula(a, b, c):
    aTob = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    bToc = math.sqrt((b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2)
    cToa = math.sqrt((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2)
    return aTob, bToc, cToa

def sidecheck(a, b, c) -> bool:
    return a + b > c and a + c > b and b + c > a

def inputTaker():
    global coordinate1, coordinate2, coordinate3
    coordinate1 = tuple(map(float, input("Coordinate of point 1 (x, y): ").split(',')))
    coordinate2 = tuple(map(float, input("Coordinate of point 2 (x, y): ").split(',')))
    coordinate3 = tuple(map(float, input("Coordinate of point 3 (x, y): ").split(',')))

inputTaker()

sides = distanceFormula(coordinate1, coordinate2, coordinate3)

if not linecheck(coordinate1, coordinate2, coordinate3) or not sidecheck(*sides):
    print("Not a triangle")
else: 
    print("Yes triangle")
