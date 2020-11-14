#!/usr/bin/env python

import matplotlib.pyplot as plt

x = range(0, 11, 1)
x2 = [x * x for x in x]
x3 = [x * x * x for x in x]

print(list(x))
print(x2)
print(x3)

fig = plt.figure(figsize=(8, 8))

ax = plt.subplot(1, 2, 1)
ax.plot(x, x2, label='quadratic')
ax.plot(x, x3, label='cubic')
ax.legend(loc='upper right')
ax.set_title('Polynomials')
fig.savefig('loss.png')
