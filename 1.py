a = [1, 2, 3, 4]
b = (5, 6, 7)
c = {8, 9, 10}

for i in a:
    if i in b:
        c.add(i)

d = [x * 2 for x in c]

print(d)
