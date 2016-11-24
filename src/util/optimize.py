"""Functions for sparse Levenberg-Marquard optimization.

sparseLM provides interface for sparse Levenberg-Marquard
optimization.

Copyright 2016 Markus Oberweger, ICG,
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
import scipy
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2016, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


def sparseLM(x0, func, fjaco, ferr, fcallback=None, lambda0=1e-2, eps_grad=1e-3, eps_param=1e-3, eps_cost=1e-3,
             eps_improv=1e-2, eps_es=1e10, max_iter=None, Lp=11, Lm=9, retall=False, disp=1, damping='eye',
             args=()):
    """
    Levenberg-Marquard optimization, see: http://people.duke.edu/~hpgavin/ce281/lm.pdf
    :param x0: initial estimate
    :param func: function callback that returns cost
    :param fjaco: function callback that returns Jacobian matrix
    :param ferr: function callback that returns per equation error
    :param fcallback: function callback that is called every iteration
    :param lambda0: initial lambda
    :param eps_grad: gradient convergence threshold
    :param eps_param: parameter convergence threshold
    :param eps_cost: cost convergence treshold
    :param eps_improv: improvement threshold
    :param eps_es: early stopping threshold, stop when objective decreased by this factor
    :param max_iter: maximum number of iterations
    :param Lp: increase factor of lambda
    :param Lm: decrease factor of lambda
    :param retall: returns list, including cost over iterations, otherwise only optimized values
    :param disp: display intermediate information
    :param damping: select damping range: eye, max, diag
    :param args: additional arguments passed to func, fjaco, ferr, fcallback, etc
    :return: optimized values
    """

    errs = []
    x_intermed = []

    x = x0.copy()
    lambdai = lambda0
    i = 0
    if max_iter is None:
        max_iter = numpy.inf
    while i < max_iter:
        func_x = func(x, *args)
        if disp:
            print "Error {} at iteration {}, lambdai {}".format(func_x, i, lambdai)

        # log errors
        if len(errs) <= i:
            errs.append(func_x)
        else:
            errs[i - 1] = func_x

        # get parts of equation
        j_mtx = fjaco(x, *args).tocsr()
        jtj = j_mtx.T.dot(j_mtx)
        dy = ferr(x, *args)
        rhs = j_mtx.T.dot(dy)

        # solve system
        # (J^T*J + lambda*I)*db = J^T*dy
        if damping == 'eye':
            diag = sps.eye(jtj.shape[0])
        elif damping == 'diag':
            diag = sps.spdiags(jtj.diagonal(), 0, jtj.shape[0], jtj.shape[1])
        elif damping == 'max':
            diag = numpy.max(jtj.diagonal())*sps.eye(jtj.shape[0])
        else:
            raise NotImplementedError("!")
        h = linsolve.spsolve(jtj + lambdai * diag, rhs)

        # calculate update
        func_xh = func(x + h, *args)
        rho_h = (func_x - func_xh) / (numpy.sum(h*(lambdai * diag * h + rhs)))
        if disp:
            print "rho={}, fun delta={}".format(rho_h, func_x - func_xh)

        if rho_h > eps_improv:
            x += h
            lambdai = max(lambdai / Lm, 1e-8)
            if fcallback is not None:
                fcallback(i, x)
            x_intermed.append(x.copy())
            i += 1
        else:
            lambdai = min(lambdai * Lp, 1e8)

            # test convergence, test separately so this is not a full iteration
            if lambdai >= 1e8:
                if disp:
                    print "Convergence lambdai {} >= {}".format(lambdai, 1e8)
                break

        if errs[0] / func_x > eps_es:
            if disp:
                print "Early stopping objective {}/{} > {}".format(errs[0], func_x, eps_es)
            break

        if numpy.abs(rhs).max() < eps_grad:
            if disp:
                print "Convergence gradient {} < {}".format(numpy.abs(rhs).max(), eps_grad)
            break

        if numpy.nanmax(numpy.abs(h / x)) < eps_param:
            if disp:
                print "Convergence parameters {} < {}".format(numpy.abs(h / x).max(), eps_param)
            break

        if func_x < eps_cost:
            if disp:
                print "Convergence cost {} < {}".format(func_x, eps_cost)
            break

    if i == max_iter:
        if disp:
            print "Maximum number of iteration {} reached!".format(i)

    if retall:
        return [x, errs, x_intermed]
    else:
        return x

