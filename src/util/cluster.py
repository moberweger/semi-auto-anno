"""Functions for clustering.

submodularClusterILP provides interface for submodular clustering
using Integer Linear Program, with solvers Gurobi and lpsolve.
submodularClusterGreedy provides interface for submodular clustering
using greedy approach.

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
import progressbar as pb

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2016, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


def submodularClusterGreedy(W, Rmax, Nmax=numpy.inf, budget=numpy.inf, tol=1e-6, useCB=False, sset=None):
    """
    Greedy submodular clustering of Krause and Golovin: Submodular Function Maximization
    :param W: pairwise similarity matrix
    :param Rmax: maximum radius
    :param Nmax: maxumum number of cluster centers
    :param budget: maximum budget C(sset) < budget
    :param sset: start set
    :return: list of reference frames
    """

    rng = numpy.random.RandomState(23455)

    if sset is None:
        sset = []
    if not isinstance(sset, list):
        raise TypeError("sset must be list or None")

    n = W.shape[0]
    V = numpy.arange(n).tolist()
    cache = numpy.inf*numpy.ones((1, n))  # cache is minimum distance of sample to closest sample in A

    def F(A, s):
        # submodular function, number of frames within the chosen distance to at least one of the frames in A
        # use cache from previous sample
        return numpy.sum(numpy.min(numpy.concatenate([W[s, :].reshape(1, n), cache], axis=0), axis=0) < Rmax)

    C = numpy.ones((n,))
    deltas = numpy.inf*numpy.ones((n,))  # initialize optimistically

    curVal = 0
    curCost = 0  # no cost

    evalNum = []
    scores = []  # keep track of statistics

    pbar = pb.ProgressBar(maxval=(Nmax if not numpy.isinf(Nmax) else W.shape[1]),
                          widgets=["Clustering", pb.Percentage(), pb.Bar()])
    pbar.start()
    i = -1
    while True:
        i += 1
        bestimprov = 0
        evalNum.append(0)
        scores.append(0)

        # init cache
        if len(sset) > 0:
            cache = numpy.min(numpy.concatenate([W[sset, :].reshape(len(sset), n), cache], axis=0), axis=0)[None, :]

        deltas[curCost+C > budget] = -numpy.inf  # cannot afford
        order = numpy.argsort(deltas)[::-1]

        # Now let's lazily update the improvements
        for test in order.tolist():
            if deltas[test] >= bestimprov:  # test could be a potential best choice
                evalNum[i] += 1
                improv = F(sset, V[test]) - curVal
                if useCB:
                    improv = improv/C[test]
                deltas[test] = improv
                bestimprov = max(bestimprov, improv)
            elif deltas[test] > -numpy.inf:
                break

        # find best delta, random permutation of ties
        # argmax = numpy.argmax(deltas)
        argmax = numpy.lexsort((rng.rand(deltas.size), deltas))[-1]
        if deltas[argmax] > tol:  # nontrivial improvement by adding argmax
            sset.append(V[argmax])
            # update cache with new sample
            cache = numpy.min(numpy.concatenate([W[sset, :].reshape(len(sset), n), cache], axis=0), axis=0)[None, :]
            if useCB:  # need to account for cost-benefit ratio
                curVal = curVal + deltas[argmax]*C[argmax]
            else:
                curVal = curVal + deltas[argmax]
            curCost = curCost + C[argmax]
            scores[i] = curVal
        else:
            break

        # got enough frames
        if len(sset) >= Nmax:
            break

        pbar.update(len(sset))

    pbar.finish()
    return numpy.asarray(sorted(sset))

