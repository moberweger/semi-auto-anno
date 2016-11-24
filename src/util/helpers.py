"""
This is the file for diverse helper functions.

Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of SemiAutoAnno.

SemiAutoAnno is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SemiAutoAnno is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SemiAutoAnno.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


def gaussian_kernel(kernel_shape, sigma=None):
    """
    Get 2D Gaussian kernel
    :param kernel_shape: kernel size
    :param sigma: sigma of Gaussian distribution
    :return: 2D Gaussian kernel
    """
    kern = numpy.zeros((kernel_shape, kernel_shape), dtype='float32')

    # get sigma from kernel size
    if sigma is None:
        sigma = 0.3*((kernel_shape-1.)*0.5 - 1.) + 0.8

    def gauss(x, y, s):
        Z = 2. * numpy.pi * s ** 2.
        return 1. / Z * numpy.exp(-(x ** 2. + y ** 2.) / (2. * s ** 2.))

    mid = numpy.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            kern[i, j] = gauss(i - mid, j - mid, sigma)

    return kern / numpy.sum(kern)

