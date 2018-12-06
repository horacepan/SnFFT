SnFFT
===

This is a proof-of-concept implementation of Clausen/Baum's FFT for the symmetric group.

For a thorough introduction to FFTs over the symmetric group, check out the following useful references:
* [Fast Fourier Transforms for the Symmetric Group: Theory and Implementation](http://www.ams.org/journals/mcom/1993-61-204/S0025-5718-1993-1192969-X/S0025-5718-1993-1192969-X.pdf), Michael Clausen and Ulrich Baum (1993)
* [Efficient computation of the Fourier transform on finite groups](https://www.researchgate.net/publication/303494890_Efficient_computation_of_the_Fourier_transform_on_finite_groups), Persi Diaconis, Daniel Rockmore (1990)

## Requirements
* numpy
* python3 (nothing in here absolutely requires python3 but most of the printing uses string [formatting](https://docs.python.org/3.5/library/functions.html#format) which is slightly different in 3)
