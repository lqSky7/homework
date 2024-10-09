d = {3: 30, 1: 10, 2: 20}
a = dict(sorted(d.items(), key=lambda x: x[1]))
b = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
print(a, b)