w1 = [-1, -1, 1, 1]
a1 = [-0.5, 0.5]

w2 = [1, -1]
a2 = [0.5]


for x1l, x1r in ((1, 0), (1,1), (0, 0), (0, 1)):
    x2l, x2r =  1 if x1l * w1[0] + x1r * w1[2]+ a1[0] > 0  else 0, 1 if x1l * w1[1] + x1r * w1[3] + a1[1] > 0 else 0
    x3 = 1 if x2l * w2[0] + x2r * w2[1] + a2[0] > 0 else 0
    print(x3)
