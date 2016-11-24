"""Provides a class that retrieves hand constraints for different
datasets and annotations.

HandConstraints provides interface for retrieving hand constraints.

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

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class HandConstraints(object):
    """
    Class for modelling hand constraints
    """

    def __init__(self, num_joints):
        """
        Constructor
        """
        self.num_joints = num_joints

        # Posebit threshold
        self.pb_thresh = 10.  # mm

        # joint offset, must be this smaller than depth
        self.joint_off = []

    def jointLength(self, j0, j1):
        return numpy.sqrt(numpy.square(j0 - j1).sum())

    def jointLengths(self, joints):
        dists = []
        for p in self.hc_pairs:
            dists.append(self.jointLength(joints[p[0]], joints[p[1]]))

        return dists

    def jointRanges(self, joints):
        assert len(joints.shape) == 3  # we need array (samples, joints, 3)
        dists = numpy.zeros((joints.shape[0], len(self.lu_pairs)))
        for i in xrange(joints.shape[0]):
            for ip, p in enumerate(self.lu_pairs):
                length = self.jointLength(joints[i, p[0]], joints[i, p[1]])
                dists[i, ip] = length

        maxd = dists.max(axis=0)
        mind = dists.min(axis=0)
        ret = []
        for ip, p in enumerate(self.lu_pairs):
            ret.append((max(mind[ip]*0.8, 0.), maxd[ip]*1.2))

        return ret

    def hc_projectionMat(self):
        """
        Generate a matrix that encodes the constraints
        [[1, 0, 0, ...]  x-coordinate, first joint
         [0, 1, 0, ...]  y-coordinate
         [0, 0, 1, ...]  z-coordinate
         [0, 0, 0, ...]
         ...
         [0, 0, 0, ...]
         [-1, 0, 0, ...]  x-coordinate, second joint
         [0, -1, 0, ...]  y-coordinate
         [0, 0, -1, ...]] z-coordinate
        :return: matrix 3*numJoints x 3*numConstraints
        """
        M = numpy.zeros((3 * self.num_joints, 3 * len(self.hc_pairs)), dtype='float32')

        ip = 0
        for p in self.hc_pairs:
            # x coordinate
            M[3 * p[0] + 0, ip] = 1
            M[3 * p[1] + 0, ip] = -1
            # y coordinate
            ip += 1
            M[3 * p[0] + 1, ip] = 1
            M[3 * p[1] + 1, ip] = -1
            # z coordinate
            ip += 1
            M[3 * p[0] + 2, ip] = 1
            M[3 * p[1] + 2, ip] = -1
            ip += 1

        return M

    def lu_projectionMat(self):
        """
        Generate a matrix that encodes the constraints
        [[1, 0, 0, ...]  x-coordinate, first joint
         [0, 1, 0, ...]  y-coordinate
         [0, 0, 1, ...]  z-coordinate
         [0, 0, 0, ...]
         ...
         [0, 0, 0, ...]
         [-1, 0, 0, ...]  x-coordinate, second joint
         [0, -1, 0, ...]  y-coordinate
         [0, 0, -1, ...]] z-coordinate
        :return: matrix 3*numJoints x 3*numConstraints
        """
        M = numpy.zeros((3 * self.num_joints, 3 * len(self.lu_pairs)), dtype='float32')

        ip = 0
        for p in self.lu_pairs:
            # x coordinate
            M[3 * p[0] + 0, ip] = 1
            M[3 * p[1] + 0, ip] = -1
            # y coordinate
            ip += 1
            M[3 * p[0] + 1, ip] = 1
            M[3 * p[1] + 1, ip] = -1
            # z coordinate
            ip += 1
            M[3 * p[0] + 2, ip] = 1
            M[3 * p[1] + 2, ip] = -1
            ip += 1

        return M

    def zz_projectionMat(self):
        """
        Generate a matrix that encodes the constraints
        [[1, 0, 0, ...]  x-coordinate, first joint
         [0, 1, 0, ...]  y-coordinate
         [0, 0, 1, ...]  z-coordinate
         [0, 0, 0, ...]
         ...
         [0, 0, 0, ...]
         [-1, 0, 0, ...]  x-coordinate, second joint
         [0, -1, 0, ...]  y-coordinate
         [0, 0, -1, ...]] z-coordinate
        :return: matrix 3*numJoints x 3*numConstraints
        """
        pair_v1M = numpy.zeros((3 * self.num_joints, 3 * len(self.zz_pairs)), dtype='float32')
        pair_v2M = numpy.zeros((3 * self.num_joints, 3 * len(self.zz_pairs)), dtype='float32')

        ip = 0
        for p in self.zz_pairs:
            # x coordinate
            pair_v1M[3 * p[0][0] + 0, ip] = 1
            pair_v1M[3 * p[0][1] + 0, ip] = -1
            pair_v2M[3 * p[1][0] + 0, ip] = 1
            pair_v2M[3 * p[1][1] + 0, ip] = -1
            # y coordinate
            ip += 1
            pair_v1M[3 * p[0][0] + 1, ip] = 1
            pair_v1M[3 * p[0][1] + 1, ip] = -1
            pair_v2M[3 * p[1][0] + 1, ip] = 1
            pair_v2M[3 * p[1][1] + 1, ip] = -1
            # z coordinate
            ip += 1
            pair_v1M[3 * p[0][0] + 2, ip] = 1
            pair_v1M[3 * p[0][1] + 2, ip] = -1
            pair_v2M[3 * p[1][0] + 2, ip] = 1
            pair_v2M[3 * p[1][1] + 2, ip] = -1
            ip += 1

        return pair_v1M, pair_v2M

    def getTemporalBreaks(self):
        """
        Breaks in sequence from i to i+1
        :return: list of indices
        """
        self.checkBreaks()

        return self.temporalBreaks

    def getSequenceBreaks(self):
        """
        Breaks in sequence from i to i+1
        :return: list of indices
        """
        self.checkBreaks()

        return self.sequenceBreaks

    def getHCBreaks(self):
        """
        Breaks in hard constraints from i to i+1
        :return: list of indices
        """
        self.checkBreaks()

        return self.hc_breaks

    def checkBreaks(self):
        """
        Check sequence, hc and temporal breaks, all must be congruent
        :return: None
        """

        # all hc breaks must be included in temporal an sequence breaks
        for hcb in self.hc_breaks:
            if hcb not in self.sequenceBreaks:
                raise ValueError("HC break {} definde, but not in sequence breaks {}!".format(hcb, self.sequenceBreaks))

            if hcb not in self.temporalBreaks:
                raise ValueError("HC break {} definde, but not in temporal breaks {}!".format(hcb, self.temporalBreaks))

        # all sequence breaks must be in temporal breaks
        for sb in self.sequenceBreaks:
            if sb not in self.temporalBreaks:
                raise ValueError("Sequence break {} definde, but not in temporal breaks {}!".format(sb, self.temporalBreaks))


class Blender2HandConstraints(HandConstraints):
    def __init__(self, seq):
        super(Blender2HandConstraints, self).__init__(24)

        if not isinstance(seq, list):
            raise ValueError("Parameter person must be list!")

        if len(seq) > 1:
            raise NotImplementedError("Not supported!")

        self.posebits = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (12, 13),
                         (14, 15), (15, 16), (16, 17), (17, 18), (19, 20), (20, 21), (21, 22), (22, 23)]

        self.hc_pairs = [(0, 1), (1, 2), (2, 3),
                         (4, 5), (5, 6), (6, 7), (7, 8),
                         (9, 10), (10, 11), (11, 12), (12, 13),
                         (14, 15), (15, 16), (16, 17), (17, 18),
                         (19, 20), (20, 21), (21, 22), (22, 23),
                         (0, 4), (4, 9), (9, 14), (14, 19)]  # almost all joints are constraint

        self.lu_pairs = []  # pairs only constrained by a range by lower and upper bounds

        # zig-zag constraint
        self.zz_pairs = [((1, 0), (2, 1)), ((2, 1), (3, 2)),
                         ((5, 4), (6, 5)), ((6, 5), (7, 6)), ((7, 6), (8, 7)),
                         ((10, 9), (11, 10)), ((11, 10), (12, 11)), ((12, 11), (13, 12)),
                         ((15, 14), (16, 15)), ((16, 15), (17, 16)), ((17, 16), (18, 17)),
                         ((20, 19), (21, 20)), ((21, 20), (22, 21)), ((22, 21), (23, 22))]
        self.zz_thresh = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # breaks in the sequence, eg due to pause in acquiring, important for temporal constraints
        # got by thresholding |pi-pi+1|^2 > 20000
        temporalBreaks = {'hpseq': [957], 'hpseq_loop_mv': [3039]}
        self.temporalBreaks = temporalBreaks[seq[0]]

        # breaks in the sequence for hard constraints, eg person changed, important for constraint handling
        # got by thresholding d(Li)-d(Li+1) > 1
        hc_breaks = {'hpseq': [957], 'hpseq_loop_mv': [3039]}
        self.hc_breaks = hc_breaks[seq[0]]

        # mark different recording sequences
        sequenceBreaks = {'hpseq': [957], 'hpseq_loop_mv': [3039]}
        self.sequenceBreaks = sequenceBreaks[seq[0]]

        # finger tips are allowed to be on the surface
        self.finger_tips = [3, 8, 13, 18, 23]

        # joint offset, must be this smaller than depth
        self.joint_off = [(20, ), (10, ), (5, ), (3, ),
                          (15, ), (5, ), (5, ), (5, ), (3, ),
                          (20, ), (7, ), (5, ), (5, ), (1, ),
                          (15, ), (7, ), (5, ), (5, ), (3, ),
                          (15, ), (4, ), (4, ), (4, ), (2, )]

        # bone length
        self.boneLength = numpy.asarray([[34.185734, 41.417736, 22.82596, 79.337524, 29.080118, 26.109291, 22.746597,
                                          74.122597, 36.783268, 28.834019, 24.202969, 66.111115, 32.468117, 27.345839,
                                          24.003115, 56.063339, 24.817657, 16.407534, 17.916492, 16.825773, 12.676471,
                                          15.136428, 8.8747139]])

        self.boneRanges = numpy.asarray([[[]]])

        self.noisePairs = [((0, 1), (1, 0)),
                         ((0, 1), (1, 2)),
                         ((1, 2), (2, 3)),
                         ((3, 2), (2, 3)),

                         ((5, 4), (4, 5)),
                         ((4, 5), (5, 6)),
                         ((5, 6), (6, 7)),
                         ((6, 7), (7, 8)),
                         ((8, 7), (7, 8)),

                         ((10, 9), (9, 10)),
                         ((9, 10), (10, 11)),
                         ((10, 11), (11, 12)),
                         ((11, 12), (12, 13)),
                         ((13, 12), (12, 13)),

                         ((15, 14), (14, 15)),
                         ((14, 15), (15, 16)),
                         ((15, 16), (16, 17)),
                         ((16, 17), (17, 18)),
                         ((18, 17), (17, 18)),

                         ((20, 19), (19, 20)),
                         ((19, 20), (20, 21)),
                         ((20, 21), (21, 22)),
                         ((21, 22), (22, 23)),
                         ((23, 22), (22, 23))]

