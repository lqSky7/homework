t = (1, 2, 3, 2, 4, 5, 3, 6, 1)
r = [i for i in set(t) if t.count(i) > 1]
print(r)