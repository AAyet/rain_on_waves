# Impact of rain-induced bubbles on wave dissipation

This code has been used for the paper Restrepo et al. 2021 NPG. Please cite the paper when using the code.

The repository contains:
- air_waves.ipnyb, a notebook used to generate the figures of the paper
- drop.npy, a polynomial fit the the data of Medwin et al. 1992 (their figure 10). It is also figure 2 of Restrepo et al. 2021. It should be called as "np.polyval(np.load('drop.npy'), d)" to get the probability of generating a bubble from a rain drop of diameter 'd'.
- a package "rain" in the bin folder, which allows computation of the volume of air entrained by bubbles generated by rain. The main function in this package is "N" which computes the density of bubbles generated by rain drops of radius "r" for a given rain rate 'R'.
