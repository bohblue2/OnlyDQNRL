import numpy as np

a = [False, False, True, False, False]
b = [1,2,3,4,5]

x = [1,2,3,4,5,6,7,8,9]
print([x if x % 2 != 0 else x * 100 for x in range(1,10)])


rr = [1,2,3,4]
qq = [[1,2],[3,4],[4,5],[5,6]]
dd = [False, False, True, False]

print([q if d else r+0.99*np.array(q) for r, q, d in zip(rr, qq, dd)])