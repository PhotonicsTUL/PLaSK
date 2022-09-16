#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('bands4.out', unpack=True)

plt.figure()
plt.plot(data[0], np.sqrt(np.sum(data[1:3,:]**2, 0)), 'k--', lw=0.7)

try:
    reference = np.loadtxt('bands6.dat', unpack=True)
except IOError:
    pass
else:
    plt.plot(reference[0], reference[3], '.', color='0.7')

plt.plot(data[0], data[3], '.', color='maroon', ms=2)

plt.xticks([0., np.pi, 2*np.pi, np.pi*(2+2**0.5)], ['$\\Gamma$', 'X', 'M', '$\\Gamma$'])
plt.xlim(0., np.pi*(2+2**0.5))
plt.grid(axis='x')

plt.ylabel("$\\omega/c$")

plt.tight_layout()
# plt.window_title('Photonic Bands')

plt.savefig('bands4.png')

plt.show()

