"""Semiautomatic annotation method.

SemiAutoAnno provides classes for creating annotations for
3D pose of articulated objects.

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
import os
import scipy
import scipy.sparse as sps
import time
import gc
import copy
import sys
import itertools
import cv2
import progressbar as pb
from sklearn.metrics.pairwise import pairwise_distances
from util.helpers import gaussian_kernel
from util.optimize import sparseLM
from util.cluster import submodularClusterGreedy

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2016, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class SemiAutoAnno:
    """Class for sample regression using minimal number of samples"""
    def __init__(self, eval_prefix, eval_params, Di, Di_off3D, Di_trans2D, Di_scale, jointLength, jointBounds, li,
                 subset_idxs, proj, cae_path, lengthConstraint, lenghtConstraintIdx, boundConstraint,
                 boundConstraintIdx, posebitIdx, pb_all, pb_thresh, temporalBreaks, hcBreaks, sequenceBreaks,
                 zz_pairs, zz_thresh, zz_pairs_v1M, zz_pairs_v2M, tips_idx,
                 lambdaW, lambdaM, lambdaP, lambdaR, lambdaMu, lambdaTh, muLagr,
                 ref_lambdaP, ref_muLagr, init_lambdaP, init_muLagr, importer, hc, depth_files,
                 li_visiblemask=None, li_posebits=None, normalizeRi=None, useCache=True, normZeroOne=False,
                 gt3D=None, hpe=None, debugPrint=False):

        self.rng = numpy.random.RandomState(23455)
        self.useCache = useCache
        self.debugPrint = debugPrint
        self.detailEvalThresh = 10000
        self.cam_proj = proj  # 3D->2D camera projection matrix
        self.gt3D = gt3D
        self.hpe = hpe
        self.hc = hc
        self.importer = importer
        self.depthFiles = depth_files
        self.Di = Di
        self.Di_off3D = Di_off3D
        self.Di_trans2D = Di_trans2D
        self.Di_scale = Di_scale

        self.eval_prefix = eval_prefix
        self.eval_params = eval_params
        if 'optimize_Ri' in self.eval_params:
            self.optimize_Ri = self.eval_params['optimize_Ri']
        else:
            raise NotImplementedError("Missing parameter 'optimize_Ri'")
        if 'optimize_bonelength' in self.eval_params:
            self.optimize_bonelength = self.eval_params['optimize_bonelength']
        else:
            raise NotImplementedError("Missing parameter 'optimize_bonelength'")
        self.eps_finished = 1e-3
        self.snapshot_iter = 10
        self.normZeroOne = normZeroOne
        self.tips = tips_idx  # finger tips are allowed to be on the surface
        if 'joint_eps' in self.eval_params:
            self.jointEps = self.eval_params['joint_eps']  # visible joints must stay within +/- eps to initialization
        else:
            raise NotImplementedError("Missing parameter 'joint_eps'")
        if 'joint_off' in self.eval_params:
            self.jointOff = self.eval_params['joint_off']  # all joints must be smaller than depth from depth map
            if isinstance(self.jointOff, list):
                if len(self.jointOff) == 1:
                    self.jointOff = self.jointOff[0]
                elif len(self.jointOff) != li.shape[1]:
                    raise ValueError("length of offset {} must fit number of joints {}".format(len(self.jointOff),
                                                                                               li.shape[1]))
        else:
            raise NotImplementedError("Missing parameter 'joint_off'")

        self.numSamples = Di.shape[0]
        self.numJoints = li.shape[1]
        self.numVar = self.numSamples*self.numJoints*3
        self.imgSize = Di.shape[2]*Di.shape[3]

        # calculate embedding from data
        if self.useCache and os.path.isfile("{}/Ri_{}.npy".format(self.eval_prefix, self.eval_params['ref_descriptor'])):
            self.Ri = numpy.load("{}/Ri_{}.npy".format(self.eval_prefix, self.eval_params['ref_descriptor']))
            if self.Ri.shape[0] != self.numSamples:
                # recalculate, cache outdated
                print "Recalculate cache. Cache has {} and samples are {}.".format(self.Ri.shape[0], self.numSamples)
                if 'ref_descriptor' in self.eval_params:
                    if self.eval_params['ref_descriptor'] == 'hog_msk':
                        self.Ri, _ = self.getImageDescriptors_HOG(self.Di, useMask=False, doNormalize=False)
                    elif self.eval_params['ref_descriptor'] == 'hog':
                        self.Ri, _ = self.getImageDescriptors_HOG(self.Di, useMask=False, doNormalize=True)
                    else:
                        print "WARNING: Cache unknown for parameter: "+self.eval_params['ref_descriptor']
                        self.Ri = numpy.asarray((1, 1))
                else:
                    raise NotImplementedError("Missing parameter 'ref_descriptor'")
                numpy.save("{}/Ri_{}.npy".format(self.eval_prefix, self.eval_params['ref_descriptor']), self.Ri)
        else:
            print "Running cache..."
            if 'ref_descriptor' in self.eval_params:
                if self.eval_params['ref_descriptor'] == 'hog_msk':
                    self.Ri, _ = self.getImageDescriptors_HOG(self.Di, useMask=False, doNormalize=False)
                elif self.eval_params['ref_descriptor'] == 'hog':
                    self.Ri, _ = self.getImageDescriptors_HOG(self.Di, useMask=False, doNormalize=True)
                else:
                    print "WARNING: Cache unknown for parameter: "+self.eval_params['ref_descriptor']
                    self.Ri = numpy.asarray((1, 1))
            else:
                raise NotImplementedError("Missing parameter 'ref_descriptor'")
            numpy.save("{}/Ri_{}.npy".format(self.eval_prefix, self.eval_params['ref_descriptor']), self.Ri)

        print "NNZ Ri: ", numpy.count_nonzero(self.Ri)/float(numpy.prod(self.Ri.shape))

        if 'corr_patch_size' in self.eval_params:
            self.corrPatchSize = self.eval_params['corr_patch_size']
        else:
            raise NotImplementedError("Missing parameter 'corr_patch_size'")
        self.corrWeighted = False
        if 'corr_method' in self.eval_params:
            self.corrMethod = self.eval_params['corr_method']
        else:
            raise NotImplementedError("Missing parameter 'corr_method'")

        self.lambdaW = lambdaW  # weight regularization
        self.lambdaM = lambdaM  # chain constraint weight
        self.lambdaP = lambdaP  # projection weight
        self.lambdaR = lambdaR  # regression weight
        self.lambdaMu = lambdaMu  # reconstruction weight
        self.lambdaTh = lambdaTh  # embedding weight
        self.muLagr = muLagr  # mu of penalty method
        self.ref_muLagr = ref_muLagr  # mu of penalty method for reference optimization
        self.ref_lambdaP = ref_lambdaP  # projection weight for reference optimization
        self.init_muLagr = init_muLagr  # mu of penalty method for initial optimization
        self.init_lambdaP = init_lambdaP  # projection weight for initial optimization

        # length constraints construction matrix of +/- 1
        self.lenghtConstraintIdx = lenghtConstraintIdx
        self.lenghtConstraint = lengthConstraint  # lC specifies constant joint distances Xi*lC - Xk*lC == (0, ... 0)
        self.numLengthConstraints = self.numSamples*len(self.lenghtConstraintIdx)
        self.lenghtConstraintS = numpy.zeros((self.lenghtConstraint.shape[1], self.lenghtConstraint.shape[1]//3),
                                             dtype='float32')
        for ip in xrange(self.lenghtConstraint.shape[1]//3):
            self.lenghtConstraintS[3*ip:3*(ip+1), ip] = 1

        # bound constraints construction matrix of +/- 1
        self.boundConstraintIdx = boundConstraintIdx
        self.boundConstraint = boundConstraint  # bC specifies bound joint distances Xi*lC - Xk*lC == (0, ... 0)
        self.numBoundConstraints = self.numSamples*len(self.boundConstraintIdx)
        self.boundConstraintS = numpy.zeros((self.boundConstraint.shape[1], self.boundConstraint.shape[1]//3),
                                            dtype='float32')
        for ip in xrange(self.boundConstraint.shape[1]//3):
            self.boundConstraintS[3*ip:3*(ip+1), ip] = 1

        # list of indices of breaks in the training sequence, important for temporal constraint handling
        self.temporalBreaks = temporalBreaks
        self.tempConstraintIdx0 = []  # indices of Li
        self.tempConstraintIdx1 = []  # indices of Li+1
        for i in xrange(self.numSamples-1):
            if i not in self.temporalBreaks:
                self.tempConstraintIdx0.append(i)
                self.tempConstraintIdx1.append(i+1)
        self.numTempConstraints = len(self.tempConstraintIdx0)*self.numJoints

        # list of breaks in sequence
        self.sequenceBreaks = sequenceBreaks

        # list of breaks in hard constraints
        self.jointLength = jointLength
        self.jointBounds = jointBounds
        self.hcBreaks = hcBreaks
        self.hcConstraintIdx0 = []  # indices of Li
        self.hcConstraintIdx1 = []  # indices of Lj
        for i in xrange(self.numSamples-1):
            if i not in self.hcBreaks:
                self.hcConstraintIdx0.append(i)
                self.hcConstraintIdx1.append(i+1)

        # zig-zag constraints
        self.zz_pairs = zz_pairs
        self.zz_thresh = zz_thresh
        self.zz_pairs_v1M = zz_pairs_v1M
        self.zz_pairs_v2M = zz_pairs_v2M
        self.zz_pairS = numpy.zeros((self.zz_pairs_v1M.shape[1], self.zz_pairs_v1M.shape[1]//3), dtype='float32')
        for ip in xrange(self.zz_pairs_v1M.shape[1]//3):
            self.zz_pairS[3*ip:3*(ip+1), ip] = 1

        if len(subset_idxs) == 0:
            subset_idxs = self.selectReferenceFrames()
            self.numSubset = len(subset_idxs)
            self.subset_idxs = numpy.asarray(sorted(subset_idxs))
            self.evalReferenceFrameSelection()
            print "Got {} reference frames".format(len(self.subset_idxs))
            print self.subset_idxs.tolist()
            thefile = open("{}/ref_idxs_{}_{}_{}.txt".format(self.eval_prefix,
                                                             self.eval_params['ref_descriptor'],
                                                             self.eval_params['ref_cluster'],
                                                             self.eval_params['ref_threshold']), 'w+')
            for item in self.subset_idxs.tolist():
                thefile.write("{}, ".format(item))
            thefile.close()
            thefile = open("{}/subset_files_{}_{}_{}.txt".format(self.eval_prefix,
                                                                 self.eval_params['ref_descriptor'],
                                                                 self.eval_params['ref_cluster'],
                                                                 self.eval_params['ref_threshold']), 'w+')
            for item in self.subset_idxs.tolist():
                thefile.write("{}\n".format(depth_files[item]))
            thefile.close()
            raise UserWarning("Rerun with given set of reference frames!")
        self.numSubset = len(subset_idxs)
        self.subset_idxs = numpy.asarray(subset_idxs)

        if li_visiblemask is None:
            # zero means: optimize me or not visible
            self.li_visiblemask = numpy.zeros((self.numSubset, self.numJoints))
        else:
            self.li_visiblemask = li_visiblemask

        self.pb_thresh = pb_thresh
        self.posebits = pb_all
        if li_posebits is None or len(li_posebits) == 0:
            # there are no posebits to help
            self.li_posebits = [[]]*self.numSubset
            self.li_posebitIdx = [[]]*self.numSubset
        else:
            self.li_posebits = li_posebits
            self.li_posebitIdx = posebitIdx

        # normalize data, normalizing pose is too dangerous, only few samples!
        if normalizeRi == "ZMUV":
            # zero mean unit variance
            self.mRi = self.Ri.mean(axis=0)
            self.stdRi = self.Ri.std(axis=0)
        elif normalizeRi == "ZO":
            # zero-one
            self.mRi = 0.
            self.stdRi = self.Ri.max()
        elif normalizeRi is None:
            self.mRi = 0.
            self.stdRi = 1.
        else:
            raise RuntimeWarning("Unknown normalization of Ri: "+str(normalizeRi))

        self.Ri -= self.mRi
        self.Ri /= self.stdRi

        self.isli2D = (li.shape[2] == 2)
        self.li = li.reshape((li.shape[0], -1))
        if self.isli2D:
            if self.useCache and os.path.isfile("{}/li3D_aug_{}.npy".format(self.eval_prefix,
                                                                            self.eval_params['ref_optimization'])):
                self.li3D_aug = numpy.load("{}/li3D_aug_{}.npy".format(self.eval_prefix,
                                                                       self.eval_params['ref_optimization']))
                if self.li3D_aug.shape[0] != len(self.subset_idxs):
                    # recalculate, cache outdated
                    print "Recalculate cache. Cache has {} and samples are {}.".format(self.li3D_aug.shape[0],
                                                                                       len(self.subset_idxs))
                    self.li3D_aug = self.augmentLi3D(self.li, self.subset_idxs)
                    numpy.save("{}/li3D_aug_{}.npy".format(self.eval_prefix, self.eval_params['ref_optimization']),
                               self.li3D_aug)
            else:
                print "Running cache..."
                self.li3D_aug = self.augmentLi3D(self.li, self.subset_idxs)
                numpy.save("{}/li3D_aug_{}.npy".format(self.eval_prefix, self.eval_params['ref_optimization']),
                           self.li3D_aug)
        else:
            self.li3D_aug = self.li

        # sanity checks
        assert numpy.all(numpy.asarray(self.subset_idxs) < self.numSamples), \
            "{} < {}".format(self.subset_idxs, self.numSamples)
        assert numpy.all(numpy.asarray(self.sequenceBreaks) <= self.numSamples), \
            "{} <= {}".format(self.sequenceBreaks, self.numSamples)
        assert numpy.all(numpy.asarray(self.hcBreaks) <= self.numSamples), \
            "{} <= {}".format(self.hcBreaks, self.numSamples)
        assert numpy.all(numpy.asarray(self.temporalBreaks) <= self.numSamples), \
            "{} <= {}".format(self.temporalBreaks, self.numSamples)
        assert self.li_visiblemask.shape[0] == self.numSubset, \
            "{} == {}".format(self.li_visiblemask.shape[0], self.numSubset)
        assert len(self.li_posebits) == self.numSubset, \
            "{} == {}".format(self.li_posebits, self.numSubset)
        assert len(self.li_posebitIdx) == self.numSubset, \
            "{} == {}".format(self.li_posebitIdx, self.numSubset)
        assert self.li3D_aug.shape[0] == self.numSubset, \
            "{} == {}".format(self.li3D_aug.shape[0], self.numSubset)
        assert len(self.hcBreaks) == self.jointLength.shape[0], \
            "{} == {}".format(len(self.hcBreaks), self.jointLength.shape[0])
        assert len(self.hcBreaks) == self.jointBounds.shape[0], \
            "{} == {}".format(len(self.hcBreaks), self.jointBounds.shape[0])

        print "#images: {}, #ref: {}, #joints: {}".format(self.numSamples, self.numSubset, self.numJoints)

        # keep copies of original versions
        self.orig_subset_idxs = numpy.asarray(subset_idxs).copy()
        self.orig_li = self.li.copy()
        self.orig_li_visiblemask = self.li_visiblemask.copy()
        self.orig_li_posebits = copy.deepcopy(self.li_posebits)
        self.orig_li_posebitIdx = copy.deepcopy(self.li_posebitIdx)
        self.orig_li3D_aug = self.li3D_aug.copy()

    def augmentLi3D(self, li2D, idxs, dptinit=None):
        """
        Augment the given 2D positions of li by using depth of image and CoM for projection from 2D -> 3D
        :param li2D: 2D annotations in normalized coordinates
        :param idxs: indices of the annotations
        :param dptinit: use provided depth for initialization in mm, otherwise augment from depth map
        :return: augmented 3D positions
        """

        if not isinstance(idxs, numpy.ndarray):
            idxs = numpy.asarray([idxs])

        assert li2D.shape[0] == len(idxs), "{}, {}".format(li2D.shape[0], len(idxs))
        if dptinit is not None:
            assert li2D.shape[0] == dptinit.shape[0] and li2D.shape[1] == dptinit.shape[1]

        # depth for 2D coordinates from depth images
        li_denorm = (li2D * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
        aug_li_img2D = numpy.concatenate([li_denorm.reshape((len(idxs), self.numJoints, 2)),
                                          numpy.zeros((len(idxs), self.numJoints, 1))], axis=2)
        # project from normalized 2D global 2D
        invM = numpy.zeros_like(self.Di_trans2D[idxs])
        for i in xrange(len(idxs)):
            invM[i] = numpy.linalg.inv(self.Di_trans2D[idxs[i]])
        li_img2D_hom = numpy.concatenate([li_denorm.reshape((len(idxs), self.numJoints, 2)),
                                          numpy.ones((len(idxs), self.numJoints, 1))], axis=2)
        li_glob2D = numpy.einsum('ijk,ikl->ijl', li_img2D_hom, invM)
        li_glob2D = li_glob2D[:, :, 0:2] / li_glob2D[:, :, 2][:, :, None]
        for i in xrange(len(idxs)):
            # use provided initialization
            if dptinit is not None:
                aug_li_img2D[i, :, 2] = dptinit[i, :]
            else:
                dm = self.importer.loadDepthMap(self.depthFiles[idxs[i]])
                y = numpy.rint(li_glob2D[i, :, 1]).astype(int).clip(0, dm.shape[0]-1)
                x = numpy.rint(li_glob2D[i, :, 0]).astype(int).clip(0, dm.shape[1]-1)
                if False:
                    # median within 3x3 mask more robust to noise
                    dpt = dm[y-1:y+2, x-1:x+2].reshape(self.numJoints, 9)
                    dpt = numpy.median(dpt[numpy.bitwise_and(dpt >= (self.Di_off3D[idxs[i], 2] - self.Di_scale[idxs[i]]),
                                                             dpt <= (self.Di_off3D[idxs[i], 2] + self.Di_scale[idxs[i]]))], axis=1)
                    # no valid dpt sample in mask or directly over background, deal with this later
                    dpt[numpy.bitwise_or(numpy.isnan(dpt), numpy.isclose(dm[y, x], self.importer.getDepthMapNV()))] = self.importer.getDepthMapNV()
                else:
                    # only take valid samples, rest is handled later
                    aug_li_img2D[i, :, 2] = dm[y, x]
                    msk = numpy.bitwise_or(dm[y, x] > (self.Di_off3D[idxs[i], 2] + self.Di_scale[idxs[i]]),
                                           dm[y, x] < (self.Di_off3D[idxs[i], 2] - self.Di_scale[idxs[i]]))
                    aug_li_img2D[i, msk, 2] = self.importer.getDepthMapNV()
                # set to default if at back of box
                msk = numpy.isclose(aug_li_img2D[i, :, 2], self.importer.getDepthMapNV())
                if numpy.any(msk):
                    if True:
                        # find closest depth from depth map for each unknown depth
                        for idx in numpy.flatnonzero(msk.ravel()):
                            validmsk = numpy.bitwise_and(dm >= (self.Di_off3D[idxs[i], 2] - self.Di_scale[idxs[i]]),
                                                         dm <= (self.Di_off3D[idxs[i], 2] + self.Di_scale[idxs[i]]))
                            pts = numpy.asarray(numpy.where(validmsk)).transpose()
                            deltas = pts - li_glob2D[i, idx, [1, 0]]
                            dist = numpy.einsum('ij,ij->i', deltas, deltas)
                            pos = numpy.argmin(dist)
                            pmin = numpy.min(dist)
                            if numpy.sqrt(pmin) > 7.:
                                # if the pixel is too far away it can be very wrong
                                # assign median depth to unknown depth
                                aug_li_img2D[i, idx, 2] = numpy.median(aug_li_img2D[i, numpy.bitwise_not(msk), 2])
                            else:
                                aug_li_img2D[i, idx, 2] = dm[pts[pos][0], pts[pos][1]]
                    else:
                        # assign median depth to unknown depth
                        aug_li_img2D[i, msk, 2] = numpy.median(aug_li_img2D[i, numpy.bitwise_not(msk), 2])
                # add joint offset
                # aug_li_img2D[i, ~msk, 2] += self.jointOff
                # threshold to box
                aug_li_img2D[i, :, 2] = aug_li_img2D[i, :, 2].clip(self.Di_off3D[idxs[i], 2] - self.Di_scale[idxs[i]],
                                                                   self.Di_off3D[idxs[i], 2] + self.Di_scale[idxs[i]])

        aug_li_img2D_norm = aug_li_img2D
        aug_li_img2D_norm[:, :, 0] = (aug_li_img2D[:, :, 0] - (self.Di.shape[3]/2.)) / (self.Di.shape[3]/2.)
        aug_li_img2D_norm[:, :, 1] = (aug_li_img2D[:, :, 1] - (self.Di.shape[3]/2.)) / (self.Di.shape[3]/2.)
        aug_li_crop3D = self.project2Dto3D(aug_li_img2D_norm, idxs)

        # save unoptimized joint positions
        numpy.save("{}/li2D_init.npy".format(self.eval_prefix), li2D)
        numpy.save("{}/li3D_init.npy".format(self.eval_prefix), aug_li_crop3D)

        if 'ref_optimization' in self.eval_params:
            if self.eval_params['ref_optimization'] == 'SLSQP':
                aug_li_crop3D = self.optimizeReferenceFramesLi_SLSQP(aug_li_crop3D, idxs)
            else:
                raise NotImplementedError("Unknown parameter: "+self.eval_params['ref_optimization'])
        else:
            raise NotImplementedError("Missing parameter 'ref_optimization'")

        return aug_li_crop3D.reshape((len(idxs), self.numJoints, 3))

    def project3Dto2D(self, Li, idxs):
        """
        Project 3D point to 2D
        :param Li: joints in normalized 3D
        :param idxs: frames specified by subset
        :return: 2D points, in normalized 2D coordinates
        """

        if not isinstance(idxs, numpy.ndarray):
            idxs = numpy.asarray([idxs])

        # 3D -> 2D projection also shift by M to cropped window
        Li_glob3D = (numpy.reshape(Li, (len(idxs), self.numJoints, 3))*self.Di_scale[idxs][:, None, None]+self.Di_off3D[idxs][:, None, :]).reshape((len(idxs)*self.numJoints, 3))
        Li_glob3D_hom = numpy.concatenate([Li_glob3D, numpy.ones((len(idxs)*self.numJoints, 1), dtype='float32')], axis=1)
        Li_glob2D_hom = numpy.dot(Li_glob3D_hom, self.cam_proj.T)
        Li_glob2D = (Li_glob2D_hom[:, 0:3] / Li_glob2D_hom[:, 3][:, None]).reshape((len(idxs), self.numJoints, 3))
        Li_img2D_hom = numpy.einsum('ijk,ikl->ijl', Li_glob2D, self.Di_trans2D[idxs])
        Li_img2D = (Li_img2D_hom[:, :, 0:2] / Li_img2D_hom[:, :, 2][:, :, None]).reshape((len(idxs), self.numJoints*2))
        Li_img2Dcrop = (Li_img2D - (self.Di.shape[3]/2.)) / (self.Di.shape[3]/2.)
        return Li_img2Dcrop

    def project2Dto3D(self, li, idxs):
        """
        Project 2D point to 3D, therefore we also need depth
        :param li: joints in normalized 2D AND z in mm
        :param idxs: frames specified by subset
        :return: 3D points, in normalized 3D coordinates
        """

        if not isinstance(idxs, numpy.ndarray):
            idxs = numpy.asarray([idxs])

        invproj = numpy.zeros((4, 4), numpy.float32)
        invproj[0, 0] = 1./self.cam_proj[0, 0]
        invproj[1, 1] = 1./self.cam_proj[1, 1]
        invproj[2, 2] = 1.
        invproj[0, 3] = -self.cam_proj[0, 2]/self.cam_proj[0, 0]
        invproj[1, 3] = -self.cam_proj[1, 2]/self.cam_proj[1, 1]
        invproj[3, 2] = 1.

        invM = numpy.zeros_like(self.Di_trans2D[idxs])
        for i in xrange(len(idxs)):
            invM[i] = numpy.linalg.inv(self.Di_trans2D[idxs[i]])

        # project from 2D -> 3D
        li = li.reshape((len(idxs), self.numJoints, 3))
        li[:, :, 0] = (li[:, :, 0] * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
        li[:, :, 1] = (li[:, :, 1] * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
        li_img2D_hom = numpy.concatenate([li[:, :, 0:2], numpy.ones((len(idxs), self.numJoints, 1))], axis=2)
        li_glob2D = numpy.einsum('ijk,ikl->ijl', li_img2D_hom, invM)
        li_glob2D = li_glob2D[:, :, 0:2] / li_glob2D[:, :, 2][:, :, None]
        li_glob2D_hom = numpy.concatenate([li_glob2D[:, :, 0:2], numpy.ones((len(idxs), self.numJoints, 2))], axis=2)
        li_glob2D_hom[:, :, 2] = li[:, :, 2]  # add z for projection into 3D
        li_glob3D_hom = numpy.dot(li_glob2D_hom.reshape((len(idxs)*self.numJoints, 4)), invproj.T)
        li_glob3D = li_glob3D_hom[:, 0:3] * li_glob3D_hom[:, 3][:, None]
        li_glob3D[:, 2] /= li_glob3D_hom[:, 3]
        li_crop3D = (li_glob3D.reshape((len(idxs), self.numJoints, 3)) - self.Di_off3D[idxs][:, None, :]) / self.Di_scale[idxs][:, None, None]

        return li_crop3D.reshape((len(idxs), self.numJoints, 3))

    def fitTracking(self, LiInit=None, num_iter=50, useLagrange=False):
        """
        Fit data with linear regression and alternate optimization
        Use penalty method or Lagrange multiplier for constraint handling
        :param LiInit: initialization of Li for "warm-start"
        :param num_iter: number of iterations of the alternate optimization
        :param useLagrange: use Augmented Lagrangian or only penalty method
        :return: weight parameters
        """

        import matplotlib.pyplot as plt
        import theano
        import theano.tensor as T

        if LiInit is None:
            Li = self.rng.randn(self.numSamples, self.numJoints * 3) * 0.01
        else:
            Li = LiInit.copy()

        if 'eval_refonly' in self.eval_params:
            if self.eval_params['eval_refonly'] is True:
                return [0, Li, Li, 0, []]
        else:
            raise NotImplementedError("Missing parameter 'eval_refonly'")

        corrMaps_k = numpy.zeros((self.numSamples, self.numJoints, self.Di.shape[2], self.Di.shape[3]), dtype='float32')
        corrMaps_kp1 = numpy.zeros((self.numSamples, self.numJoints, self.Di.shape[2], self.Di.shape[3]), dtype='float32')
        weights_k = numpy.zeros((self.numSamples, ))
        weights_kp1 = numpy.zeros((self.numSamples, ))

        if LiInit is None:
            # initial estimate of Li given weights
            # Li = self.predictLinear(weights, feat=Ri)
            # common pose, does not work well, as this is a local minimum
            # Li = numpy.tile(self.li[0], (K.shape[0], 1))
            if self.useCache and os.path.isfile("{}/Li_init_{}.npy".format(self.eval_prefix,
                                                                           self.eval_params['init_method'])):
                Li = numpy.load("{}/Li_init_{}.npy".format(self.eval_prefix, self.eval_params['init_method']))
                if os.path.isfile("{}/Li_init_{}_corrected.npy".format(self.eval_prefix, self.eval_params['init_method'])):
                    corrected = numpy.load("{}/Li_init_{}_corrected.npy".format(self.eval_prefix, self.eval_params['init_method'])).tolist()
                else:
                    corrected = []
            else:
                if self.isli2D:
                    if 'init_method' in self.eval_params:
                        if self.eval_params['init_method'] == 'closest':
                            Li, corrected = self.initClosestReference3D_maps(self.li3D_aug)
                            Li = Li.reshape((self.numSamples, -1))
                        else:
                            raise NotImplementedError("Unknown parameter: "+self.eval_params['init_method'])
                    else:
                        raise NotImplementedError("Missing parameter 'init_method'")
                else:
                    if 'init_method' in self.eval_params:
                        if self.eval_params['init_method'] == 'closest':
                            Li, corrected = self.initClosestReference3D_maps(self.li)
                            Li = Li.reshape((self.numSamples, -1))
                        else:
                            raise NotImplementedError("Unknown parameter: "+self.eval_params['init_method'])
                    else:
                        raise NotImplementedError("Missing parameter 'init_method'")
                numpy.save("{}/Li_init_{}.npy".format(self.eval_prefix, self.eval_params['init_method']), Li)
                numpy.save("{}/Li_init_{}_corrected.npy".format(self.eval_prefix, self.eval_params['init_method']), numpy.asarray(corrected))

        if 'eval_initonly' in self.eval_params and self.eval_params['eval_initonly'] is True:
            return [0, Li, Li, 0, []]

        Li_init = Li.copy()

        lambdaLagr = self.rng.randn(self.numLengthConstraints) * 0.0
        tsli = theano.shared(self.li.astype('float32'), name='li', borrow=True)
        tsLi = theano.shared(Li.astype('float32'), name='Li', borrow=True)
        tsMuLagr = theano.shared(numpy.cast['float32'](self.muLagr), name='muLagr', borrow=True)
        tsCam = theano.shared(self.cam_proj.astype('float32'), name='camera', borrow=True)
        tsScale = theano.shared(self.Di_scale.astype('float32'), name='scale', borrow=True)
        tsOff3D = theano.shared(self.Di_off3D.astype('float32'), name='off3D', borrow=True)
        tsTrans2D = theano.shared(self.Di_trans2D.astype('float32'), name='trans2D', borrow=True)
        tsLagr = theano.shared(lambdaLagr.astype('float32'), name='lambdaLagr', borrow=True)
        tsJL = theano.shared(self.jointLength.astype('float32'), name='joint length', borrow=True)
        tsCorr_k = theano.shared(corrMaps_k.astype('float32'), name='corr_k', borrow=True)
        tsCorr_kp1 = theano.shared(corrMaps_kp1.astype('float32'), name='corr_kp1', borrow=True)
        tsCorr_w_k = theano.shared(weights_k.astype('float32'), name='corr_w_k', borrow=True)
        tsCorr_w_kp1 = theano.shared(weights_kp1.astype('float32'), name='corr_w_kp1', borrow=True)

        # cost function
        cost = 0
        # temporal constraint
        if 'global_tempconstraints' in self.eval_params:
            if self.eval_params['global_tempconstraints'] == 'local':
                cost += self.lambdaM/self.numTempConstraints*T.sum(T.sqr(tsLi[self.tempConstraintIdx0] - tsLi[self.tempConstraintIdx1]).sum(axis=1))
            elif self.eval_params['global_tempconstraints'] == 'global':
                Li_glob = T.reshape(tsLi, (self.numSamples, self.numJoints, 3))*tsScale.dimshuffle(0, 'x', 'x')+tsOff3D.dimshuffle(0, 'x', 1)
                cost += self.lambdaM/self.numTempConstraints*T.sum(T.sqr((Li_glob[self.tempConstraintIdx0] - Li_glob[self.tempConstraintIdx1])/tsScale[self.tempConstraintIdx0].dimshuffle(0, 'x', 'x')).sum(axis=1))
            elif self.eval_params['global_tempconstraints'] == 'none':
                cost += 0
            else:
                raise NotImplementedError("Unknown parameter: "+self.eval_params['global_tempconstraints'])
        else:
            raise NotImplementedError("Missing parameter 'global_tempconstraints'")

        if self.isli2D:
            # 3D -> 2D projection also shift by M to cropped window
            Li_subset = tsLi[self.subset_idxs]
            Li_subset_glob3D = (T.reshape(Li_subset, (self.numSubset, self.numJoints, 3))*tsScale[self.subset_idxs].dimshuffle(0, 'x', 'x')+tsOff3D[self.subset_idxs].dimshuffle(0, 'x', 1)).reshape((self.numSubset*self.numJoints, 3))
            Li_subset_glob3D_hom = T.concatenate([Li_subset_glob3D, T.ones((self.numSubset*self.numJoints, 1), dtype='float32')], axis=1)
            Li_subset_glob2D_hom = T.dot(Li_subset_glob3D_hom, tsCam.T)
            Li_subset_glob2D_hom = T.set_subtensor(Li_subset_glob2D_hom[:, 0], T.true_div(Li_subset_glob2D_hom[:, 0], Li_subset_glob2D_hom[:, 3]))
            Li_subset_glob2D_hom = T.set_subtensor(Li_subset_glob2D_hom[:, 1], T.true_div(Li_subset_glob2D_hom[:, 1], Li_subset_glob2D_hom[:, 3]))
            Li_subset_glob2D = T.set_subtensor(Li_subset_glob2D_hom[:, 2], T.true_div(Li_subset_glob2D_hom[:, 2], Li_subset_glob2D_hom[:, 3]))
            Li_subset_glob2D = Li_subset_glob2D[:, 0:3].reshape((self.numSubset, self.numJoints, 3))
            Li_subset_img2D_hom = T.batched_dot(Li_subset_glob2D, tsTrans2D[self.subset_idxs])
            Li_subset_img2D_hom=T.set_subtensor(Li_subset_img2D_hom[:, :, 0], T.true_div(Li_subset_img2D_hom[:, :, 0], Li_subset_img2D_hom[:, :, 2]))
            Li_subset_img2D_hom=T.set_subtensor(Li_subset_img2D_hom[:, :, 1], T.true_div(Li_subset_img2D_hom[:, :, 1], Li_subset_img2D_hom[:, :, 2]))
            Li_subset_img2D = T.set_subtensor(Li_subset_img2D_hom[:, :, 2], T.true_div(Li_subset_img2D_hom[:, :, 2], Li_subset_img2D_hom[:, :, 2]))
            Li_subset_img2D = Li_subset_img2D[:, :, 0:2].reshape((self.numSubset, self.numJoints*2))
            Li_subset_img2Dcrop = (Li_subset_img2D - (self.Di.shape[3]/2.)) / (self.Di.shape[3]/2.)
            cost += self.lambdaP/self.numJoints/self.numSubset*T.sum(T.sqr(Li_subset_img2Dcrop - tsli).sum(axis=1))  # 3D to 2D projection
        else:
            cost += self.lambdaP/self.numJoints/self.numSubset*T.sum(T.sqr(tsLi[self.subset_idxs] - tsli).sum(axis=1))  # li in 3D

        # 3D -> 2D projection also shift by M to cropped window
        Li_glob3D = (T.reshape(tsLi, (self.numSamples, self.numJoints, 3))*tsScale.dimshuffle(0, 'x', 'x')+tsOff3D.dimshuffle(0, 'x', 1)).reshape((self.numSamples*self.numJoints, 3))
        Li_glob3D_hom = T.concatenate([Li_glob3D, T.ones((self.numSamples*self.numJoints, 1), dtype='float32')], axis=1)
        Li_glob2D_hom = T.dot(Li_glob3D_hom, tsCam.T)
        Li_glob2D_hom = T.set_subtensor(Li_glob2D_hom[:, 0], T.true_div(Li_glob2D_hom[:, 0], Li_glob2D_hom[:, 3]))
        Li_glob2D_hom = T.set_subtensor(Li_glob2D_hom[:, 1], T.true_div(Li_glob2D_hom[:, 1], Li_glob2D_hom[:, 3]))
        Li_glob2D = T.set_subtensor(Li_glob2D_hom[:, 2], T.true_div(Li_glob2D_hom[:, 2], Li_glob2D_hom[:, 3]))
        Li_glob2D = Li_glob2D[:, 0:3].reshape((self.numSamples, self.numJoints, 3))
        Li_img2D_hom = T.batched_dot(Li_glob2D, tsTrans2D)
        Li_img2D_hom=T.set_subtensor(Li_img2D_hom[:, :, 0], T.true_div(Li_img2D_hom[:, :, 0], Li_img2D_hom[:, :, 2]))
        Li_img2D_hom=T.set_subtensor(Li_img2D_hom[:, :, 1], T.true_div(Li_img2D_hom[:, :, 1], Li_img2D_hom[:, :, 2]))
        Li_img2D = T.set_subtensor(Li_img2D_hom[:, :, 2], T.true_div(Li_img2D_hom[:, :, 2], Li_img2D_hom[:, :, 2]))
        Li_img2D = Li_img2D[:, :, 0:2].reshape((self.numSamples, self.numJoints, 2))
        # bilinear interpolation on correlation maps
        i, j = numpy.ogrid[0:self.numSamples, 0:self.numJoints]
        x1 = T.cast(T.floor(Li_img2D[:, :, 0]), 'int64').clip(0, self.Di.shape[3]-2)
        y1 = T.cast(T.floor(Li_img2D[:, :, 1]), 'int64').clip(0, self.Di.shape[2]-2)
        x2 = T.cast(T.ceil(Li_img2D[:, :, 0]), 'int64').clip(0, self.Di.shape[3]-2)
        y2 = T.cast(T.ceil(Li_img2D[:, :, 1]), 'int64').clip(0, self.Di.shape[2]-2)
        # div by zero error if x1 = x2 = x if x in Z
        x2 = T.switch(T.eq(x1, x2), x1+1, x2)
        y2 = T.switch(T.eq(y1, y2), y1+1, y2)
        x = Li_img2D[:, :, 0]
        y = Li_img2D[:, :, 1]
        fQ11_k = tsCorr_k[i, j, y1, x1]
        fQ21_k = tsCorr_k[i, j, y1, x2]
        fQ12_k = tsCorr_k[i, j, y2, x1]
        fQ22_k = tsCorr_k[i, j, y2, x2]
        val_k = 1./(x2-x1)/(y2-y1)*(fQ11_k*(x2-x)*(y2-y) + fQ21_k*(x-x1)*(y2-y) + fQ12_k*(x2-x)*(y-y1) + fQ22_k*(x-x1)*(y-y1))
        fQ11_kp1 = tsCorr_kp1[i, j, y1, x1]
        fQ21_kp1 = tsCorr_kp1[i, j, y1, x2]
        fQ12_kp1 = tsCorr_kp1[i, j, y2, x1]
        fQ22_kp1 = tsCorr_kp1[i, j, y2, x2]
        val_kp1 = 1./(x2-x1)/(y2-y1)*(fQ11_kp1*(x2-x)*(y2-y) + fQ21_kp1*(x-x1)*(y2-y) + fQ12_kp1*(x2-x)*(y-y1) + fQ22_kp1*(x-x1)*(y-y1))
        cost += self.lambdaR/self.numJoints/self.numSamples*T.sum(T.sqr(val_k*tsCorr_w_k.dimshuffle(0, 'x') + val_kp1*tsCorr_w_kp1.dimshuffle(0, 'x')))
        # cost += self.lambdaR/self.numJoints/self.numSamples*T.sum(T.sqr(tsCorr_k[i, j, Li_img2D[:, :, 1], Li_img2D[:, :, 0]]*tsCorr_w_k.dimshuffle(0, 'x') + tsCorr_kp1[i, j, Li_img2D[:, :, 1], Li_img2D[:, :, 0]]*tsCorr_w_kp1.dimshuffle(0, 'x')))
        # constraint with joint length
        di = T.dot(T.sqr(T.dot(tsLi, self.lenghtConstraint)), self.lenghtConstraintS)
        eq = (di - T.sqr(T.extra_ops.repeat(tsJL, numpy.ediff1d(numpy.insert(self.hcBreaks, 0, -1)), axis=0))).flatten()
        cost += tsMuLagr/self.numLengthConstraints*T.sum(T.sqr(eq))  # Lagrangian of constraints, penalty method
        if useLagrange:
            cost -= T.sum(T.dot(tsLagr.T, eq))  # Lagrangian of constraints, augmented lagrangian method

        eq_constr = T.sum(T.sqr(eq))  # equality constraint

        print "compiling functions..."
        givens = []
        fun_cost = theano.function([], cost, givens=givens, mode='FAST_RUN', on_unused_input='warn')

        if self.optimize_bonelength:
            var = [tsLi, tsJL]
        else:
            var = [tsLi]
        grad_cost = T.grad(cost, var)
        gf_cost = theano.function([], grad_cost, givens=givens, mode='FAST_RUN', on_unused_input='warn')

        fun_eq = theano.function([], eq, givens=givens, mode='FAST_RUN', on_unused_input='warn')

        fun_eq_constr = theano.function([], eq_constr, givens=givens, mode='FAST_RUN', on_unused_input='warn')

        grad_eq_constr = T.grad(eq_constr, var)
        gf_eq_constr = theano.function([], grad_eq_constr, givens=givens, mode='FAST_RUN', on_unused_input='warn')
        print "done"

        # creates a function that computes the cost
        def train_fn(Li_val, ref_tsLi, ref_tsJL, ref_tsMuLagr):
            if self.optimize_bonelength:
                ref_tsLi.set_value(Li_val[0:self.numVar].reshape(Li.shape).astype('float32'))
                ref_tsJL.set_value(Li_val[self.numVar:].reshape(self.jointLength.shape).astype('float32'))
            else:
                ref_tsLi.set_value(Li_val.reshape(Li.shape).astype('float32'))
            return fun_cost().flatten().astype(float)

        # creates a function that computes the gradient
        def train_fn_grad(Li_val, ref_tsLi, ref_tsJL, ref_tsMuLagr):
            if self.optimize_bonelength:
                ref_tsLi.set_value(Li_val[0:self.numVar].reshape(Li.shape).astype('float32'))
                ref_tsJL.set_value(Li_val[self.numVar:].reshape(self.jointLength.shape).astype('float32'))
            else:
                ref_tsLi.set_value(Li_val.reshape(Li.shape).astype('float32'))
            return gf_cost().flatten().astype(float)

        def callback_fn(Li_val, ref_tsLi, ref_tsJL, ref_tsMuLagr):
            if self.optimize_bonelength:
                ref_tsLi.set_value(Li_val[0:self.numVar].reshape(Li.shape).astype('float32'))
                ref_tsJL.set_value(Li_val[self.numVar:].reshape(self.jointLength.shape).astype('float32'))
            else:
                ref_tsLi.set_value(Li_val.reshape(Li.shape).astype('float32'))

            # update shared parameter
            # tsMuLagr.set_value(numpy.cast['float32'](tsMuLagr.get_value()*5.), borrow=True)  # update this, mu -> inf
            print "callback cost: "+str(fun_cost())

        # creates a function that computes the equality cost
        def eq_constr_fn(Li_val, ref_tsLi, ref_tsJL, ref_tsMuLagr):
            if self.optimize_bonelength:
                ref_tsLi.set_value(Li_val[0:self.numVar].reshape(Li.shape).astype('float32'))
                ref_tsJL.set_value(Li_val[self.numVar:].reshape(self.jointLength.shape).astype('float32'))
            else:
                ref_tsLi.set_value(Li_val.reshape(Li.shape).astype('float32'))
            return fun_eq_constr().flatten().astype(float)

        # creates a function that computes the gradient of equality cost
        def eq_constr_fn_grad(Li_val, ref_tsLi, ref_tsJL, ref_tsMuLagr):
            if self.optimize_bonelength:
                ref_tsLi.set_value(Li_val[0:self.numVar].reshape(Li.shape).astype('float32'))
                ref_tsJL.set_value(Li_val[self.numVar:].reshape(self.jointLength.shape).astype('float32'))
            else:
                ref_tsLi.set_value(Li_val.reshape(Li.shape).astype('float32'))
            return gf_eq_constr().flatten().astype(float)

        #######################
        # create correlation maps
        pbar = pb.ProgressBar(maxval=self.numSamples, widgets=["Correlation maps", pb.Percentage(), pb.Bar()])
        pbar.start()
        s = 0

        if 'global_corr_ref' in self.eval_params:
            if self.eval_params['global_corr_ref'] == 'closest':
                if self.isli2D:
                    li_denorm = self.li*(self.Di.shape[3]/2.)+(self.Di.shape[3]/2.)
                else:
                    li_denorm = self.project3Dto2D(self.li, self.subset_idxs)*(self.Di.shape[3]/2.)+(self.Di.shape[3]/2.)
                # iterate all images in current sequence
                for i in xrange(self.numSamples):
                    pbar.update(i)
                    ref_idx, _, _ = self.getReferenceForSample(i, Li, doOffset=False)
                    li_idx = numpy.where(numpy.asarray(self.subset_idxs) == ref_idx)[0][0]

                    weights_k[i] = 1.
                    weights_kp1[i] = 0.

                    for j in xrange(self.numJoints):
                        y_k = numpy.floor(li_denorm[li_idx].reshape((self.numJoints, 2))[j, 1]).astype(int).clip(0, self.Di.shape[2]-1)
                        x_k = numpy.floor(li_denorm[li_idx].reshape((self.numJoints, 2))[j, 0]).astype(int).clip(0, self.Di.shape[3]-1)
                        ystart_k = y_k - self.corrPatchSize // 2
                        yend_k = ystart_k+self.corrPatchSize
                        xstart_k = x_k - self.corrPatchSize // 2
                        xend_k = xstart_k+self.corrPatchSize

                        # create template patches and pad them
                        templ_k = self.Di[ref_idx, 0, max(ystart_k, 0):min(yend_k, self.Di.shape[2]),
                                          max(xstart_k, 0):min(xend_k, self.Di.shape[3])].copy()
                        templ_k = numpy.pad(templ_k, ((abs(ystart_k)-max(ystart_k, 0), abs(yend_k)-min(yend_k, self.Di.shape[2])),
                                                      (abs(xstart_k)-max(xstart_k, 0), abs(xend_k)-min(xend_k, self.Di.shape[3]))),
                                            mode='constant', constant_values=(1.,))

                        # apply gaussian weighted template
                        if self.corrWeighted is True:
                            gw = gaussian_kernel(templ_k.shape[0])#, sigma=10.)
                            gw /= gw.max()  # normalize to 0-1
                            templ_k *= gw

                        # mask to fight background
                        msk = numpy.bitwise_and(self.Di[i, 0] < 1.-1e-4, self.Di[i, 0] > -1.+1e-4)
                        msk = scipy.ndimage.binary_erosion(msk, structure=numpy.ones((3, 3)), iterations=1)
                        msk_dt = numpy.bitwise_not(msk)
                        edt = scipy.ndimage.morphology.distance_transform_edt(msk_dt)
                        edt = cv2.normalize(edt, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                        # normalized correlation maps
                        di_pad = numpy.pad(self.Di[i, 0].astype('float32'), ((self.corrPatchSize // 2, self.corrPatchSize // 2-1),
                                                                             (self.corrPatchSize // 2, self.corrPatchSize // 2-1)),
                                           mode='constant', constant_values=(1.,))
                        corrMaps_k[i, j] = cv2.matchTemplate(di_pad, templ_k.astype('float32'), self.corrMethod)
                        corrMaps_k[i, j] *= msk.astype(corrMaps_k.dtype)
                        corrMaps_k[i, j] = cv2.normalize(corrMaps_k[i, j], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                        # correlation is high where similar, we minimize dissimilarity which is 1-corr
                        corrMaps_k[i, j] = 1. - corrMaps_k[i, j] + numpy.square(edt + 1.)*msk_dt.astype(corrMaps_k.dtype)
                        corrMaps_kp1[i, j] = 0.

                        # scipy.misc.imshow(templ_k)
                        # scipy.misc.imshow(templ_kp1)
                        # scipy.misc.imshow(corrMaps_k[i,j])
                        # scipy.misc.imshow(corrMaps_kp1[i,j])
                        # print cv2.minMaxLoc(corrMaps_k[i,j]), x_k, y_k
                        # print cv2.minMaxLoc(corrMaps_kp1[i,j]), x_kp1, y_kp1

                        # clean up
                        del templ_k
                gc.collect()
                gc.collect()
                gc.collect()
            else:
                raise NotImplementedError("Unknown parameter: "+self.eval_params['global_corr_ref'])
        else:
            raise NotImplementedError("Missing parameter 'global_corr_ref'")

        pbar.finish()

        allcosts = []
        allerrsLi = []
        allerrsLiref = []

        # monitor overall cost
        costM = self.lambdaM/self.numTempConstraints*numpy.square(Li[self.tempConstraintIdx0, :] - Li[self.tempConstraintIdx1, :]).sum()
        if self.isli2D:
            costP = self.lambdaP/self.numJoints/self.numSubset*numpy.square(self.project3Dto2D(Li[self.subset_idxs], self.subset_idxs) - self.li).sum()  # 3D to 2D projection
        else:
            costP = self.lambdaP/self.numJoints/self.numSubset*numpy.square(Li[self.subset_idxs] - self.li).sum()  # li in 3D

        Li_img2D = (self.project3Dto2D(Li, numpy.arange(self.numSamples)) * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
        Li_img2D = Li_img2D.reshape((self.numSamples, self.numJoints, 2))
        # bilinear interpolation on correlation maps
        i, j = numpy.ogrid[0:self.numSamples, 0:self.numJoints]
        x1 = numpy.floor(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
        y1 = numpy.floor(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
        x2 = numpy.ceil(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
        y2 = numpy.ceil(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
        # div by zero error if x1 = x2 = x if x in Z
        x2[numpy.where(x1 == x2)] += 1
        y2[numpy.where(y1 == y2)] += 1
        x = Li_img2D[:, :, 0]
        y = Li_img2D[:, :, 1]
        fQ11_k = corrMaps_k[i, j, y1, x1]
        fQ21_k = corrMaps_k[i, j, y1, x2]
        fQ12_k = corrMaps_k[i, j, y2, x1]
        fQ22_k = corrMaps_k[i, j, y2, x2]
        val_k = 1./(x2-x1)/(y2-y1)*(fQ11_k*(x2-x)*(y2-y) + fQ21_k*(x-x1)*(y2-y) + fQ12_k*(x2-x)*(y-y1) + fQ22_k*(x-x1)*(y-y1))
        fQ11_kp1 = corrMaps_kp1[i, j, y1, x1]
        fQ21_kp1 = corrMaps_kp1[i, j, y1, x2]
        fQ12_kp1 = corrMaps_kp1[i, j, y2, x1]
        fQ22_kp1 = corrMaps_kp1[i, j, y2, x2]
        val_kp1 = 1./(x2-x1)/(y2-y1)*(fQ11_kp1*(x2-x)*(y2-y) + fQ21_kp1*(x-x1)*(y2-y) + fQ12_kp1*(x2-x)*(y-y1) + fQ22_kp1*(x-x1)*(y-y1))
        costR = self.lambdaR/self.numJoints/self.numSamples*numpy.sum(numpy.square(val_k*weights_k[:, None] + val_kp1*weights_kp1[:, None]))
        # costR = self.lambdaR/self.numJoints/self.numSamples*numpy.sum(numpy.square(corrMaps_k[i, j, Li_img2D[:, :, 1], Li_img2D[:, :, 0]]*weights_k[:, None] + corrMaps_kp1[i, j, Li_img2D[:, :, 1], Li_img2D[:, :, 0]]*weights_kp1[:, None]))
        di = numpy.dot(numpy.square(numpy.dot(Li, self.lenghtConstraint)), self.lenghtConstraintS)
        eq = di - numpy.square(numpy.repeat(self.jointLength, numpy.ediff1d(numpy.insert(self.hcBreaks, 0, -1)), axis=0))
        costConstr = tsMuLagr.get_value()/self.numLengthConstraints*numpy.sum(numpy.square(eq))  # equality constraint

        # original cost function
        allcost = costM + costP + costR
        allcosts.append([allcost, costM, costP, costR, costConstr])
        print "ALL COSTS: "+str(allcosts[-1])+" Theano:"+str(fun_cost())
        allerrsLi.append(self.evaluateToGT(Li, numpy.arange(self.numSamples)))
        print "Errors to GT: "+str(allerrsLi[-1])
        allerrsLiref.append(self.evaluateToGT(Li, self.subset_idxs))
        print "Errors to ref: "+str(allerrsLiref[-1])

        # solve for Li
        if self.optimize_bonelength:
            x0 = numpy.concatenate([Li.flatten(), self.jointLength.flatten()], axis=0)
        else:
            x0 = Li.flatten()

        def train_fn_jaco(x, ref_tsLi, ref_tsJL, ref_tsMuLagr):
            # |y_i - f(x_i)|^2
            # J = df(x)/dx
            row = []
            col = []
            data = []
            offset = 0

            # first block diagonal
            if self.isli2D:
                for ns in xrange(0, self.numSubset):
                    for k in xrange(0, self.numJoints):
                        row.extend([ns*self.numJoints*2+k*2,
                                    ns*self.numJoints*2+k*2+1,
                                    ns*self.numJoints*2+k*2,
                                    ns*self.numJoints*2+k*2+1,
                                    ns*self.numJoints*2+k*2,
                                    ns*self.numJoints*2+k*2+1])
                        col.extend([self.subset_idxs[ns]*self.numJoints*3+k*3+0,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+0,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+1,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+1,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+2,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+2])

                        s = self.Di_scale[self.subset_idxs[ns]]
                        cx = self.Di_off3D[self.subset_idxs[ns]][0]
                        cy = self.Di_off3D[self.subset_idxs[ns]][1]
                        cz = self.Di_off3D[self.subset_idxs[ns]][2]
                        px = x[self.subset_idxs[ns]*self.numJoints*3+k*3+0]
                        py = x[self.subset_idxs[ns]*self.numJoints*3+k*3+1]
                        pz = x[self.subset_idxs[ns]*self.numJoints*3+k*3+2]
                        kx = self.Di_trans2D[self.subset_idxs[ns]][0, 0]
                        ky = self.Di_trans2D[self.subset_idxs[ns]][1, 1]
                        fx = self.cam_proj[0, 0]
                        fy = self.cam_proj[1, 1]
                        norm = (self.Di.shape[3]/2.)
                        data.extend([kx*fx*s/(norm*s*pz+norm*cz)*self.lambdaP/self.numJoints/self.numSubset, 0.,
                                     0., ky*fy*s/(norm*s*pz+norm*cz)*self.lambdaP/self.numJoints/self.numSubset,
                                     -kx*fx*(s*px+cx)*norm*s/(norm*s*pz+norm*cz)**2.*self.lambdaP/self.numJoints/self.numSubset,
                                     -ky*fy*(s*py+cy)*norm*s/(norm*s*pz+norm*cz)**2.*self.lambdaP/self.numJoints/self.numSubset])
            else:
                for ns in xrange(0, self.numSubset):
                    for k in xrange(0, self.numJoints):
                        for j in xrange(0, 3):
                            col.extend([ns*self.numJoints*3+k*3+j])
                            row.extend([ns*self.numJoints*3+k*3+j])
                        data.extend((numpy.ones((3,))*self.lambdaP/self.numJoints/self.numSubset).tolist())

            if self.isli2D:
                offset += self.numSubset*self.numJoints*2
            else:
                offset = self.numSubset*self.numJoints*3
            # correlation maps
            Li_img2D = (self.project3Dto2D(x.reshape(Li.shape), numpy.arange(self.numSamples)) * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
            Li_img2D = Li_img2D.reshape((self.numSamples, self.numJoints, 2))
            # bilinear interpolation on correlation maps
            i, j = numpy.ogrid[0:self.numSamples, 0:self.numJoints]
            x1 = numpy.floor(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
            y1 = numpy.floor(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
            x2 = numpy.ceil(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
            y2 = numpy.ceil(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
            # div by zero error if x1 = x2 = x if x in Z
            x2[numpy.where(x1 == x2)] += 1
            y2[numpy.where(y1 == y2)] += 1
            xx = Li_img2D[:, :, 0]
            yy = Li_img2D[:, :, 1]
            fQ11_k = corrMaps_k[i, j, y1, x1]
            fQ21_k = corrMaps_k[i, j, y1, x2]
            fQ12_k = corrMaps_k[i, j, y2, x1]
            fQ22_k = corrMaps_k[i, j, y2, x2]
            val_k_dx = 1./(x2-x1)/(y2-y1)*(-fQ11_k*(y2-yy) + fQ21_k*(y2-yy) - fQ12_k*(yy-y1) + fQ22_k*(yy-y1))
            val_k_dy = 1./(x2-x1)/(y2-y1)*(-fQ11_k*(x2-xx) - fQ21_k*(xx-x1) + fQ12_k*(x2-xx) + fQ22_k*(xx-x1))
            fQ11_kp1 = corrMaps_kp1[i, j, y1, x1]
            fQ21_kp1 = corrMaps_kp1[i, j, y1, x2]
            fQ12_kp1 = corrMaps_kp1[i, j, y2, x1]
            fQ22_kp1 = corrMaps_kp1[i, j, y2, x2]
            val_kp1_dx = 1./(x2-x1)/(y2-y1)*(-fQ11_kp1*(y2-yy) + fQ21_kp1*(y2-yy) - fQ12_kp1*(yy-y1) + fQ22_kp1*(yy-y1))
            val_kp1_dy = 1./(x2-x1)/(y2-y1)*(-fQ11_kp1*(x2-xx) - fQ21_kp1*(xx-x1) + fQ12_kp1*(x2-xx) + fQ22_kp1*(xx-x1))
            grad_x = val_k_dx*weights_k[:, None] + val_kp1_dx*weights_kp1[:, None]
            grad_y = val_k_dy*weights_k[:, None] + val_kp1_dy*weights_kp1[:, None]

            for ns in xrange(0, self.numSamples):
                for k in xrange(0, self.numJoints):
                    row.extend([ns*self.numJoints+k+offset,
                                ns*self.numJoints+k+offset,
                                ns*self.numJoints+k+offset])
                    col.extend([ns*self.numJoints*3+k*3+0,
                                ns*self.numJoints*3+k*3+1,
                                ns*self.numJoints*3+k*3+2])

                    s = self.Di_scale[ns]
                    cx = self.Di_off3D[ns][0]
                    cy = self.Di_off3D[ns][1]
                    cz = self.Di_off3D[ns][2]
                    px = x[ns*self.numJoints*3+k*3+0]
                    py = x[ns*self.numJoints*3+k*3+1]
                    pz = x[ns*self.numJoints*3+k*3+2]
                    kx = self.Di_trans2D[ns][0, 0]
                    ky = self.Di_trans2D[ns][1, 1]
                    fx = self.cam_proj[0, 0]
                    fy = self.cam_proj[1, 1]
                    data.extend([grad_x[ns, k] * kx*fx*s/(s*pz+cz)*self.lambdaR/self.numJoints/self.numSamples,
                                 grad_y[ns, k] * ky*fy*s/(s*pz+cz)*self.lambdaR/self.numJoints/self.numSamples,
                                 (grad_x[ns, k] * -kx*fx*(s*px+cx)*s/(s*pz+cz)**2. + grad_y[ns, k] * -ky*fy*(s*py+cy)*s/(s*pz+cz)**2.)*self.lambdaR/self.numJoints/self.numSamples])

            offset += self.numSamples*self.numJoints
            # temporal constraint, off-diagonal
            for ns in xrange(0, len(self.tempConstraintIdx0)):
                for k in xrange(0, self.numJoints):
                    for j in xrange(0, 3):
                        row.extend([ns*self.numJoints*3+k*3+j+offset, ns*self.numJoints*3+k*3+j+offset])
                        col.extend([self.tempConstraintIdx0[ns]*self.numJoints*3+k*3+j,
                                    self.tempConstraintIdx1[ns]*self.numJoints*3+k*3+j])
                        data.extend([1.*self.lambdaM/self.numTempConstraints,
                                     -1.*self.lambdaM/self.numTempConstraints])

            offset += len(self.tempConstraintIdx0)*3*self.numJoints
            # hard constraint, off-diagonal
            lag = ref_tsMuLagr.get_value()
            for ns in xrange(0, self.numSamples):
                for k in xrange(len(self.lenghtConstraintIdx)):
                    row.extend([ns*len(self.lenghtConstraintIdx)+k+offset,
                                ns*len(self.lenghtConstraintIdx)+k+offset,
                                ns*len(self.lenghtConstraintIdx)+k+offset,
                                ns*len(self.lenghtConstraintIdx)+k+offset,
                                ns*len(self.lenghtConstraintIdx)+k+offset,
                                ns*len(self.lenghtConstraintIdx)+k+offset])
                    col.extend([ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+0,
                                ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+1,
                                ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+2,
                                ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+0,
                                ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+1,
                                ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+2])
                    data.extend([2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+0] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+0])*lag/self.numLengthConstraints,
                                 2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+1] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+1])*lag/self.numLengthConstraints,
                                 2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+2] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+2])*lag/self.numLengthConstraints,
                                 2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+0] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+0])*lag/self.numLengthConstraints,
                                 2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+1] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+1])*lag/self.numLengthConstraints,
                                 2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+2] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+2])*lag/self.numLengthConstraints]) # lag[ns*len(self.lenghtConstraintIdx)+k]]
                    if self.optimize_bonelength:
                        row.append(ns*len(self.lenghtConstraintIdx)+k+offset)
                        col.append(self.numVar+self.hcIdxForFlatIdx(ns)*len(self.lenghtConstraintIdx)+k)
                        data.append(-2.*x[self.numVar+self.hcIdxForFlatIdx(ns)*len(self.lenghtConstraintIdx)+k])
            offset += self.numSamples*len(self.lenghtConstraintIdx)
            nvars = self.numSamples*self.numJoints*3
            if self.optimize_bonelength:
                nvars += len(self.lenghtConstraintIdx)*len(self.hcBreaks)
            spmat = sps.coo_matrix((data, (row, col)), shape=(offset, nvars), dtype=float)

            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.spy(spmat, marker='.', markersize=2)
            # plt.show(block=True)

            return spmat

        def train_fn_jaco_norefs(x, ref_tsLi, ref_tsJL, ref_tsMuLagr):
            # |y_i - f(x_i)|^2
            # J = df(x)/dx
            row = []
            col = []
            data = []
            offset = 0

            # first block diagonal
            if self.isli2D:
                for ns in xrange(0, self.numSubset):
                    for k in xrange(0, self.numJoints):
                        row.extend([ns*self.numJoints*2+k*2,
                                    ns*self.numJoints*2+k*2+1,
                                    ns*self.numJoints*2+k*2,
                                    ns*self.numJoints*2+k*2+1,
                                    ns*self.numJoints*2+k*2,
                                    ns*self.numJoints*2+k*2+1])
                        col.extend([self.subset_idxs[ns]*self.numJoints*3+k*3+0,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+0,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+1,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+1,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+2,
                                    self.subset_idxs[ns]*self.numJoints*3+k*3+2])

                        data.extend([0., 0., 0., 0., 0., 0.])
            else:
                for ns in xrange(0, self.numSubset):
                    for k in xrange(0, self.numJoints):
                        for j in xrange(0, 3):
                            col.extend([ns*self.numJoints*3+k*3+j])
                            row.extend([ns*self.numJoints*3+k*3+j])
                        data.extend([0., 0., 0.])

            if self.isli2D:
                offset += self.numSubset*self.numJoints*2
            else:
                offset = self.numSubset*self.numJoints*3
            # correlation maps
            Li_img2D = (self.project3Dto2D(x.reshape(Li.shape), numpy.arange(self.numSamples)) * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
            Li_img2D = Li_img2D.reshape((self.numSamples, self.numJoints, 2))
            # bilinear interpolation on correlation maps
            i, j = numpy.ogrid[0:self.numSamples, 0:self.numJoints]
            x1 = numpy.floor(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
            y1 = numpy.floor(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
            x2 = numpy.ceil(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
            y2 = numpy.ceil(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
            # div by zero error if x1 = x2 = x if x in Z
            x2[numpy.where(x1 == x2)] += 1
            y2[numpy.where(y1 == y2)] += 1
            xx = Li_img2D[:, :, 0]
            yy = Li_img2D[:, :, 1]
            fQ11_k = corrMaps_k[i, j, y1, x1]
            fQ21_k = corrMaps_k[i, j, y1, x2]
            fQ12_k = corrMaps_k[i, j, y2, x1]
            fQ22_k = corrMaps_k[i, j, y2, x2]
            val_k_dx = 1./(x2-x1)/(y2-y1)*(-fQ11_k*(y2-yy) + fQ21_k*(y2-yy) - fQ12_k*(yy-y1) + fQ22_k*(yy-y1))
            val_k_dy = 1./(x2-x1)/(y2-y1)*(-fQ11_k*(x2-xx) - fQ21_k*(xx-x1) + fQ12_k*(x2-xx) + fQ22_k*(xx-x1))
            fQ11_kp1 = corrMaps_kp1[i, j, y1, x1]
            fQ21_kp1 = corrMaps_kp1[i, j, y1, x2]
            fQ12_kp1 = corrMaps_kp1[i, j, y2, x1]
            fQ22_kp1 = corrMaps_kp1[i, j, y2, x2]
            val_kp1_dx = 1./(x2-x1)/(y2-y1)*(-fQ11_kp1*(y2-yy) + fQ21_kp1*(y2-yy) - fQ12_kp1*(yy-y1) + fQ22_kp1*(yy-y1))
            val_kp1_dy = 1./(x2-x1)/(y2-y1)*(-fQ11_kp1*(x2-xx) - fQ21_kp1*(xx-x1) + fQ12_kp1*(x2-xx) + fQ22_kp1*(xx-x1))
            grad_x = val_k_dx*weights_k[:, None] + val_kp1_dx*weights_kp1[:, None]
            grad_y = val_k_dy*weights_k[:, None] + val_kp1_dy*weights_kp1[:, None]

            for ns in xrange(0, self.numSamples):
                if ns not in self.subset_idxs:
                    for k in xrange(0, self.numJoints):
                        row.extend([ns*self.numJoints+k+offset,
                                    ns*self.numJoints+k+offset,
                                    ns*self.numJoints+k+offset])
                        col.extend([ns*self.numJoints*3+k*3+0,
                                    ns*self.numJoints*3+k*3+1,
                                    ns*self.numJoints*3+k*3+2])

                        s = self.Di_scale[ns]
                        cx = self.Di_off3D[ns][0]
                        cy = self.Di_off3D[ns][1]
                        cz = self.Di_off3D[ns][2]
                        px = x[ns*self.numJoints*3+k*3+0]
                        py = x[ns*self.numJoints*3+k*3+1]
                        pz = x[ns*self.numJoints*3+k*3+2]
                        kx = self.Di_trans2D[ns][0, 0]
                        ky = self.Di_trans2D[ns][1, 1]
                        fx = self.cam_proj[0, 0]
                        fy = self.cam_proj[1, 1]
                        data.extend([grad_x[ns, k] * kx*fx*s/(s*pz+cz)*self.lambdaR/self.numJoints/self.numSamples,
                                     grad_y[ns, k] * ky*fy*s/(s*pz+cz)*self.lambdaR/self.numJoints/self.numSamples,
                                     (grad_x[ns, k] * -kx*fx*(s*px+cx)*s/(s*pz+cz)**2. + grad_y[ns, k] * -ky*fy*(s*py+cy)*s/(s*pz+cz)**2.)*self.lambdaR/self.numJoints/self.numSamples])
                else:
                    for k in xrange(0, self.numJoints):
                        row.extend([ns*self.numJoints+k+offset,
                                    ns*self.numJoints+k+offset,
                                    ns*self.numJoints+k+offset])
                        col.extend([ns*self.numJoints*3+k*3+0,
                                    ns*self.numJoints*3+k*3+1,
                                    ns*self.numJoints*3+k*3+2])
                        data.extend([0., 0., 0.])

            offset += self.numSamples*self.numJoints
            # temporal constraint, off-diagonal
            for ns in xrange(0, len(self.tempConstraintIdx0)):
                if self.tempConstraintIdx0[ns] not in self.subset_idxs and self.tempConstraintIdx1[ns] not in self.subset_idxs:
                    for k in xrange(0, self.numJoints):
                        for j in xrange(0, 3):
                            row.extend([ns*self.numJoints*3+k*3+j+offset, ns*self.numJoints*3+k*3+j+offset])
                            col.extend([self.tempConstraintIdx0[ns]*self.numJoints*3+k*3+j,
                                        self.tempConstraintIdx1[ns]*self.numJoints*3+k*3+j])
                            data.extend([1.*self.lambdaM/self.numTempConstraints,
                                         -1.*self.lambdaM/self.numTempConstraints])
                elif self.tempConstraintIdx0[ns] not in self.subset_idxs:
                    for k in xrange(0, self.numJoints):
                        for j in xrange(0, 3):
                            row.extend([ns*self.numJoints*3+k*3+j+offset, ns*self.numJoints*3+k*3+j+offset])
                            col.extend([self.tempConstraintIdx0[ns]*self.numJoints*3+k*3+j,
                                        self.tempConstraintIdx1[ns]*self.numJoints*3+k*3+j])
                            data.extend([1.*self.lambdaM/self.numTempConstraints, 0.])
                elif self.tempConstraintIdx1[ns] not in self.subset_idxs:
                    for k in xrange(0, self.numJoints):
                        for j in xrange(0, 3):
                            row.extend([ns*self.numJoints*3+k*3+j+offset, ns*self.numJoints*3+k*3+j+offset])
                            col.extend([self.tempConstraintIdx0[ns]*self.numJoints*3+k*3+j,
                                        self.tempConstraintIdx1[ns]*self.numJoints*3+k*3+j])
                            data.extend([0., -1.*self.lambdaM/self.numTempConstraints])
                else:
                    for k in xrange(0, self.numJoints):
                        for j in xrange(0, 3):
                            row.extend([ns*self.numJoints*3+k*3+j+offset, ns*self.numJoints*3+k*3+j+offset])
                            col.extend([self.tempConstraintIdx0[ns]*self.numJoints*3+k*3+j,
                                        self.tempConstraintIdx1[ns]*self.numJoints*3+k*3+j])
                            data.extend([0., 0.])

            offset += len(self.tempConstraintIdx0)*3*self.numJoints
            # hard constraint, off-diagonal
            lag = ref_tsMuLagr.get_value()
            for ns in xrange(0, self.numSamples):
                if ns not in self.subset_idxs:
                    for k in xrange(len(self.lenghtConstraintIdx)):
                        row.extend([ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset])
                        col.extend([ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+0,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+1,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+2,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+0,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+1,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+2])
                        data.extend([2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+0] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+0])*lag/self.numLengthConstraints,
                                     2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+1] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+1])*lag/self.numLengthConstraints,
                                     2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+2] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+2])*lag/self.numLengthConstraints,
                                     2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+0] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+0])*lag/self.numLengthConstraints,
                                     2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+1] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+1])*lag/self.numLengthConstraints,
                                     2.*(x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+2] - x[ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+2])*lag/self.numLengthConstraints]) # lag[ns*len(self.lenghtConstraintIdx)+k]]
                        if self.optimize_bonelength:
                            row.append(ns*len(self.lenghtConstraintIdx)+k+offset)
                            col.append(self.numVar+self.hcIdxForFlatIdx(ns)*len(self.lenghtConstraintIdx)+k)
                            data.append(-2.*x[self.numVar+self.hcIdxForFlatIdx(ns)*len(self.lenghtConstraintIdx)+k])
                else:
                    for k in xrange(len(self.lenghtConstraintIdx)):
                        row.extend([ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset,
                                    ns*len(self.lenghtConstraintIdx)+k+offset])
                        col.extend([ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+0,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+1,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][0]*3+2,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+0,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+1,
                                    ns*self.numJoints*3+self.lenghtConstraintIdx[k][1]*3+2])
                        data.extend([0., 0., 0., 0., 0., 0.])
                        if self.optimize_bonelength:
                            row.append(ns*len(self.lenghtConstraintIdx)+k+offset)
                            col.append(self.numVar+self.hcIdxForFlatIdx(ns)*len(self.lenghtConstraintIdx)+k)
                            data.append(0)

            offset += self.numSamples*len(self.lenghtConstraintIdx)
            nvars = self.numSamples*self.numJoints*3
            if self.optimize_bonelength:
                nvars += len(self.lenghtConstraintIdx)*len(self.hcBreaks)
            spmat = sps.coo_matrix((data, (row, col)), shape=(offset, nvars), dtype=float)

            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.spy(spmat, marker='.', markersize=2)
            # plt.show(block=True)

            return spmat

        def train_fn_err(x, ref_tsLi, ref_tsJL, ref_tsMuLagr):
            # y_i - f(x_i)
            Li_img2D = (self.project3Dto2D(x[0:self.numVar].reshape(Li.shape), numpy.arange(self.numSamples)) * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
            Li_img2D = Li_img2D.reshape((self.numSamples, self.numJoints, 2))
            # bilinear interpolation on correlation maps
            i, j = numpy.ogrid[0:self.numSamples, 0:self.numJoints]
            x1 = numpy.floor(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
            y1 = numpy.floor(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
            x2 = numpy.ceil(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
            y2 = numpy.ceil(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
            # div by zero error if x1 = x2 = x if x in Z
            x2[numpy.where(x1 == x2)] += 1
            y2[numpy.where(y1 == y2)] += 1
            xx = Li_img2D[:, :, 0]
            yy = Li_img2D[:, :, 1]
            fQ11_k = corrMaps_k[i, j, y1, x1]
            fQ21_k = corrMaps_k[i, j, y1, x2]
            fQ12_k = corrMaps_k[i, j, y2, x1]
            fQ22_k = corrMaps_k[i, j, y2, x2]
            val_k = 1./(x2-x1)/(y2-y1)*(fQ11_k*(x2-xx)*(y2-yy) + fQ21_k*(xx-x1)*(y2-yy) + fQ12_k*(x2-xx)*(yy-y1) + fQ22_k*(xx-x1)*(yy-y1))
            fQ11_kp1 = corrMaps_kp1[i, j, y1, x1]
            fQ21_kp1 = corrMaps_kp1[i, j, y1, x2]
            fQ12_kp1 = corrMaps_kp1[i, j, y2, x1]
            fQ22_kp1 = corrMaps_kp1[i, j, y2, x2]
            val_kp1 = 1./(x2-x1)/(y2-y1)*(fQ11_kp1*(x2-xx)*(y2-yy) + fQ21_kp1*(xx-x1)*(y2-yy) + fQ12_kp1*(x2-xx)*(yy-y1) + fQ22_kp1*(xx-x1)*(yy-y1))

            if self.optimize_bonelength:
                if 'global_tempconstraints' in self.eval_params:
                    if self.eval_params['global_tempconstraints'] == 'local':
                        tempErr = self.lambdaM/self.numTempConstraints*(x[0:self.numVar].reshape(Li.shape)[self.tempConstraintIdx1] - x[0:self.numVar].reshape(Li.shape)[self.tempConstraintIdx0]).ravel()
                    elif self.eval_params['global_tempconstraints'] == 'global':
                        Li_glob = (x[0:self.numVar].reshape(self.numSamples, self.numJoints, 3)*self.Di_scale[:, None, None]+self.Di_off3D[:, None, :]).reshape(Li.shape)
                        tempErr = self.lambdaM/self.numTempConstraints*((Li_glob[self.tempConstraintIdx1] - Li_glob[self.tempConstraintIdx0])/self.Di_scale[self.tempConstraintIdx0, None]).ravel()
                    elif self.eval_params['global_tempconstraints'] == 'none':
                        tempErr = numpy.zeros((self.numTempConstraints, ))
                    else:
                        raise NotImplementedError("Unknown parameter: "+self.eval_params['global_tempconstraints'])
                else:
                    raise NotImplementedError("Missing parameter 'global_tempconstraints'")

                err = numpy.concatenate([self.lambdaP/self.numJoints/self.numSubset*((self.li - self.project3Dto2D(x[0:self.numVar].reshape(Li.shape)[self.subset_idxs], self.subset_idxs)).ravel()) if self.isli2D else self.lambdaP/self.numJoints/self.numSubset*((self.li - x[0:self.numVar].reshape(Li.shape)[self.subset_idxs]).ravel()),
                                         self.lambdaR/self.numJoints/self.numSamples*(-(val_k*weights_k[:, None] + val_kp1*weights_kp1[:, None]).ravel()),
                                         tempErr,
                                         ref_tsMuLagr.get_value()/self.numLengthConstraints*(numpy.square(numpy.repeat(x[self.numVar:].reshape(self.jointLength.shape), numpy.ediff1d(numpy.insert(self.hcBreaks, 0, -1)), axis=0)) - numpy.dot(numpy.square(numpy.dot(x[0:self.numVar].reshape(Li.shape), self.lenghtConstraint)), self.lenghtConstraintS)).ravel()])
            else:
                if 'global_tempconstraints' in self.eval_params:
                    if self.eval_params['global_tempconstraints'] == 'local':
                        tempErr = self.lambdaM/self.numTempConstraints*(x.reshape(Li.shape)[self.tempConstraintIdx1] - x.reshape(Li.shape)[self.tempConstraintIdx0]).ravel()
                    elif self.eval_params['global_tempconstraints'] == 'global':
                        Li_glob = (x.reshape(self.numSamples, self.numJoints, 3)*self.Di_scale[:, None, None]+self.Di_off3D[:, None, :]).reshape(Li.shape)
                        tempErr = self.lambdaM/self.numTempConstraints*((Li_glob[self.tempConstraintIdx1] - Li_glob[self.tempConstraintIdx0])/self.Di_scale[self.tempConstraintIdx0, None]).ravel()
                    elif self.eval_params['global_tempconstraints'] == 'none':
                        tempErr = numpy.zeros((self.numTempConstraints, ))
                    else:
                        raise NotImplementedError("Unknown parameter: "+self.eval_params['global_tempconstraints'])
                else:
                    raise NotImplementedError("Missing parameter 'global_tempconstraints'")

                err = numpy.concatenate([self.lambdaP/self.numJoints/self.numSubset*((self.li - self.project3Dto2D(x.reshape(Li.shape)[self.subset_idxs], self.subset_idxs)).ravel()) if self.isli2D else self.lambdaP/self.numJoints/self.numSubset*((self.li - x.reshape(Li.shape)[self.subset_idxs]).ravel()),
                                         self.lambdaR/self.numJoints/self.numSamples*(-(val_k*weights_k[:, None] + val_kp1*weights_kp1[:, None]).ravel()),
                                         tempErr,
                                         ref_tsMuLagr.get_value()/self.numLengthConstraints*(numpy.square(numpy.repeat(self.jointLength, numpy.ediff1d(numpy.insert(self.hcBreaks, 0, -1)), axis=0)) - numpy.dot(numpy.square(numpy.dot(x.reshape(Li.shape), self.lenghtConstraint)), self.lenghtConstraintS)).ravel()])
            return err

        def train_fn_err_norefs(x, ref_tsLi, ref_tsJL, ref_tsMuLagr):
            # y_i - f(x_i)
            Li_img2D = (self.project3Dto2D(x[0:self.numVar].reshape(Li.shape), numpy.arange(self.numSamples)) * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
            Li_img2D = Li_img2D.reshape((self.numSamples, self.numJoints, 2))
            # bilinear interpolation on correlation maps
            i, j = numpy.ogrid[0:self.numSamples, 0:self.numJoints]
            x1 = numpy.floor(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
            y1 = numpy.floor(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
            x2 = numpy.ceil(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
            y2 = numpy.ceil(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
            # div by zero error if x1 = x2 = x if x in Z
            x2[numpy.where(x1 == x2)] += 1
            y2[numpy.where(y1 == y2)] += 1
            xx = Li_img2D[:, :, 0]
            yy = Li_img2D[:, :, 1]
            fQ11_k = corrMaps_k[i, j, y1, x1]
            fQ21_k = corrMaps_k[i, j, y1, x2]
            fQ12_k = corrMaps_k[i, j, y2, x1]
            fQ22_k = corrMaps_k[i, j, y2, x2]
            val_k = 1./(x2-x1)/(y2-y1)*(fQ11_k*(x2-xx)*(y2-yy) + fQ21_k*(xx-x1)*(y2-yy) + fQ12_k*(x2-xx)*(yy-y1) + fQ22_k*(xx-x1)*(yy-y1))
            fQ11_kp1 = corrMaps_kp1[i, j, y1, x1]
            fQ21_kp1 = corrMaps_kp1[i, j, y1, x2]
            fQ12_kp1 = corrMaps_kp1[i, j, y2, x1]
            fQ22_kp1 = corrMaps_kp1[i, j, y2, x2]
            val_kp1 = 1./(x2-x1)/(y2-y1)*(fQ11_kp1*(x2-xx)*(y2-yy) + fQ21_kp1*(xx-x1)*(y2-yy) + fQ12_kp1*(x2-xx)*(yy-y1) + fQ22_kp1*(xx-x1)*(yy-y1))

            maskR = numpy.ones((self.numSamples, self.numJoints))
            maskR[self.subset_idxs, :] = 0.
            maskT = numpy.ones((len(self.tempConstraintIdx1)*self.numJoints*3, ))
            maskHC = numpy.ones((self.numSamples, len(self.lenghtConstraintIdx)))
            maskHC[self.subset_idxs, :] = 0.

            if self.optimize_bonelength:
                if 'global_tempconstraints' in self.eval_params:
                    if self.eval_params['global_tempconstraints'] == 'local':
                        tempErr = self.lambdaM/self.numTempConstraints*(x[0:self.numVar].reshape(Li.shape)[self.tempConstraintIdx1] - x[0:self.numVar].reshape(Li.shape)[self.tempConstraintIdx0]).ravel()
                    elif self.eval_params['global_tempconstraints'] == 'global':
                        Li_glob = (x[0:self.numVar].reshape(self.numSamples, self.numJoints, 3)*self.Di_scale[:, None, None]+self.Di_off3D[:, None, :]).reshape(Li.shape)
                        tempErr = self.lambdaM/self.numTempConstraints*((Li_glob[self.tempConstraintIdx1] - Li_glob[self.tempConstraintIdx0])/self.Di_scale[self.tempConstraintIdx0, None]).ravel()
                    elif self.eval_params['global_tempconstraints'] == 'none':
                        tempErr = numpy.zeros((self.numTempConstraints, ))
                    else:
                        raise NotImplementedError("Unknown parameter: "+self.eval_params['global_tempconstraints'])
                else:
                    raise NotImplementedError("Missing parameter 'global_tempconstraints'")

                err = numpy.concatenate([numpy.zeros_like(self.li).ravel() if self.isli2D else numpy.zeros_like(self.li).ravel(),
                                         maskR.ravel()*self.lambdaR/self.numJoints/self.numSamples*(-(val_k*weights_k[:, None] + val_kp1*weights_kp1[:, None]).ravel()),
                                         maskT.ravel()*tempErr,
                                         maskHC.ravel()*ref_tsMuLagr.get_value()/self.numLengthConstraints*(numpy.square(numpy.repeat(x[self.numVar:].reshape(self.jointLength.shape), numpy.ediff1d(numpy.insert(self.hcBreaks, 0, -1)), axis=0)) - numpy.dot(numpy.square(numpy.dot(x[0:self.numVar].reshape(Li.shape), self.lenghtConstraint)), self.lenghtConstraintS)).ravel()])
            else:
                if 'global_tempconstraints' in self.eval_params:
                    if self.eval_params['global_tempconstraints'] == 'local':
                        tempErr = self.lambdaM/self.numTempConstraints*(x.reshape(Li.shape)[self.tempConstraintIdx1] - x.reshape(Li.shape)[self.tempConstraintIdx0]).ravel()
                    elif self.eval_params['global_tempconstraints'] == 'global':
                        Li_glob = (x.reshape(self.numSamples, self.numJoints, 3)*self.Di_scale[:, None, None]+self.Di_off3D[:, None, :]).reshape(Li.shape)
                        tempErr = self.lambdaM/self.numTempConstraints*((Li_glob[self.tempConstraintIdx1] - Li_glob[self.tempConstraintIdx0])/self.Di_scale[self.tempConstraintIdx0, None]).ravel()
                    elif self.eval_params['global_tempconstraints'] == 'none':
                        tempErr = numpy.zeros((self.numTempConstraints, ))
                    else:
                        raise NotImplementedError("Unknown parameter: "+self.eval_params['global_tempconstraints'])
                else:
                    raise NotImplementedError("Missing parameter 'global_tempconstraints'")

                err = numpy.concatenate([numpy.zeros_like(self.li).ravel() if self.isli2D else numpy.zeros_like(self.li).ravel(),
                                         maskR.ravel()*self.lambdaR/self.numJoints/self.numSamples*(-(val_k*weights_k[:, None] + val_kp1*weights_kp1[:, None]).ravel()),
                                         maskT.ravel()*tempErr,
                                         maskHC.ravel()*ref_tsMuLagr.get_value()/self.numLengthConstraints*(numpy.square(numpy.repeat(self.jointLength, numpy.ediff1d(numpy.insert(self.hcBreaks, 0, -1)), axis=0)) - numpy.dot(numpy.square(numpy.dot(x.reshape(Li.shape), self.lenghtConstraint)), self.lenghtConstraintS)).ravel()])
            return err

        start = time.time()

        if 'global_optimize_incHC' in self.eval_params:
            if self.eval_params['global_optimize_incHC'] is True:
                best_rval = [x0]
                for mi in range(5):
                    best_rval = sparseLM(x0=best_rval[0],
                                         func=train_fn,
                                         fjaco=train_fn_jaco,
                                         ferr=train_fn_err,
                                         max_iter=100,
                                         eps_grad=1e-11,
                                         eps_param=1e-8,
                                         eps_cost=1e-8,
                                         eps_improv=1e-8,
                                         retall=True,
                                         args=(tsLi, tsJL, tsMuLagr))
                    tsMuLagr.set_value(numpy.cast['float32'](tsMuLagr.get_value()*10.))
            else:
                best_rval = sparseLM(x0=x0,
                                     func=train_fn,
                                     fjaco=train_fn_jaco,
                                     ferr=train_fn_err,
                                     max_iter=100,
                                     eps_grad=1e-11,
                                     eps_param=1e-8,
                                     eps_cost=1e-8,
                                     eps_improv=1e-8,
                                     retall=True,
                                     args=(tsLi, tsJL, tsMuLagr))
        else:
            raise NotImplementedError("Missing parameter 'global_optimize_incHC'")

        print "Took {}s for fitTracking".format(time.time()-start)

        if self.optimize_bonelength:
            Li = best_rval[0][0:self.numVar].reshape(Li.shape)
            self.jointLength = best_rval[0][self.numVar:].reshape(self.jointLength.shape)
        else:
            Li = best_rval[0].reshape(Li.shape)
        tsLi.set_value(Li.astype('float32'), borrow=True)
        tsJL.set_value(self.jointLength.astype('float32'), borrow=True)
        lambdaLagr = lambdaLagr - tsMuLagr.get_value()*fun_eq()
        tsLagr.set_value(lambdaLagr.astype('float32'), borrow=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(best_rval[1])
        ax.set_yscale('symlog')
        ax.grid('on')
        plt.show(block=False)
        fig.savefig(self.eval_prefix+'/costsLM_appearance.pdf', bbox_inches='tight')
        plt.close(fig)

        # monitor overall cost
        costM = self.lambdaM/self.numTempConstraints*numpy.square(Li[self.tempConstraintIdx0, :] - Li[self.tempConstraintIdx1, :]).sum()
        if self.isli2D:
            costP = self.lambdaP/self.numJoints/self.numSubset*numpy.square(self.project3Dto2D(Li[self.subset_idxs], self.subset_idxs) - self.li).sum()  # 3D to 2D projection
        else:
            costP = self.lambdaP/self.numJoints/self.numSubset*numpy.square(Li[self.subset_idxs] - self.li).sum()  # li in 3D

        Li_img2D = (self.project3Dto2D(Li, numpy.arange(self.numSamples)) * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
        Li_img2D = Li_img2D.reshape((self.numSamples, self.numJoints, 2))
        # bilinear interpolation on correlation maps
        i, j = numpy.ogrid[0:self.numSamples, 0:self.numJoints]
        x1 = numpy.floor(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
        y1 = numpy.floor(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
        x2 = numpy.ceil(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
        y2 = numpy.ceil(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
        # div by zero error if x1 = x2 = x if x in Z
        x2[numpy.where(x1 == x2)] += 1
        y2[numpy.where(y1 == y2)] += 1
        x = Li_img2D[:, :, 0]
        y = Li_img2D[:, :, 1]
        fQ11_k = corrMaps_k[i, j, y1, x1]
        fQ21_k = corrMaps_k[i, j, y1, x2]
        fQ12_k = corrMaps_k[i, j, y2, x1]
        fQ22_k = corrMaps_k[i, j, y2, x2]
        val_k = 1./(x2-x1)/(y2-y1)*(fQ11_k*(x2-x)*(y2-y) + fQ21_k*(x-x1)*(y2-y) + fQ12_k*(x2-x)*(y-y1) + fQ22_k*(x-x1)*(y-y1))
        fQ11_kp1 = corrMaps_kp1[i, j, y1, x1]
        fQ21_kp1 = corrMaps_kp1[i, j, y1, x2]
        fQ12_kp1 = corrMaps_kp1[i, j, y2, x1]
        fQ22_kp1 = corrMaps_kp1[i, j, y2, x2]
        val_kp1 = 1./(x2-x1)/(y2-y1)*(fQ11_kp1*(x2-x)*(y2-y) + fQ21_kp1*(x-x1)*(y2-y) + fQ12_kp1*(x2-x)*(y-y1) + fQ22_kp1*(x-x1)*(y-y1))
        costR = self.lambdaR/self.numJoints/self.numSamples*numpy.sum(numpy.square(val_k*weights_k[:, None] + val_kp1*weights_kp1[:, None]))
        # costR = self.lambdaR/self.numJoints/self.numSamples*numpy.sum(numpy.square(corrMaps_k[i, j, Li_img2D[:, :, 1], Li_img2D[:, :, 0]]*weights_k[:, None] + corrMaps_kp1[i, j, Li_img2D[:, :, 1], Li_img2D[:, :, 0]]*weights_kp1[:, None]))
        # di = numpy.dot(numpy.square(numpy.dot(Li[self.hcConstraintIdx0, :], self.lenghtConstraint)), self.lenghtConstraintS)
        # dj = numpy.dot(numpy.square(numpy.dot(Li[self.hcConstraintIdx1, :], self.lenghtConstraint)), self.lenghtConstraintS)
        # eq = numpy.sqrt(di) - numpy.sqrt(dj)
        di = numpy.dot(numpy.square(numpy.dot(Li, self.lenghtConstraint)), self.lenghtConstraintS)
        eq = di - numpy.square(numpy.repeat(self.jointLength, numpy.ediff1d(numpy.insert(self.hcBreaks, 0, -1)), axis=0))
        costConstr = tsMuLagr.get_value()/self.numLengthConstraints*numpy.sum(numpy.square(eq))  # equality constraint

        # original cost function
        allcost = costM + costP + costR
        allcosts.append([allcost, costM, costP, costR, costConstr])
        print "ALL COSTS: "+str(allcosts[-1])+" Theano:"+str(fun_cost())
        allerrsLi.append(self.evaluateToGT(Li, numpy.arange(self.numSamples)))
        print "Errors to GT: "+str(allerrsLi[-1])
        allerrsLiref.append(self.evaluateToGT(Li, self.subset_idxs))
        print "Errors to ref: "+str(allerrsLiref[-1])

        leg = ['allcost', 'costM', 'costP', 'costR', 'costConstr']
        self.plotErrors(allerrsLi, allerrsLiref, [[0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]], allcosts, leg)

        # Clean up, to free possible gpu memory for several function calls
        del tsli, tsCam, tsScale, tsOff3D, tsTrans2D, tsLagr, tsCorr_k, tsCorr_kp1, tsCorr_w_k, tsCorr_w_kp1, tsLi, tsJL, tsMuLagr
        gc.collect()
        gc.collect()
        gc.collect()

        return [0, Li, Li_init, 0, corrected]

    def optimizeReferenceFramesLi_SLSQP(self, li3D, idxs):
        """
        Optimize on the 3D positions of li before initialization, thus we ge better linear interpolation for Li
        Here we have all depth images and 2D annotations so it should be easier
        :param li3D: 3D projections of li
        :param idxs: indices of the projections
        :return: optimized 3D locations of li
        """

        import theano
        import theano.tensor as T

        if not isinstance(idxs, numpy.ndarray):
            idxs = numpy.asarray([idxs])

        if not self.isli2D:
            raise RuntimeError("Not needed!")

        orig = li3D.copy()

        jointOccupacy = 0  # 15.  # sphere that is occupied by the joint

        li3D = li3D.reshape((li3D.shape[0], -1))
        tsLi = theano.shared(li3D[0].astype('float32'), name='li3D', borrow=True)
        tsli = theano.shared(self.li[0].astype('float32'), name='li', borrow=True)
        tsMuLagr = theano.shared(numpy.cast['float32'](self.ref_muLagr), name='muLagr', borrow=True)
        tsCam = theano.shared(self.cam_proj.astype('float32'), name='camera', borrow=True)
        tsScale = theano.shared(self.Di_scale[idxs[0]].astype('float32'), name='scale', borrow=True)
        tsOff3D = theano.shared(self.Di_off3D[idxs[0]].astype('float32'), name='off3D', borrow=True)
        tsTrans2D = theano.shared(self.Di_trans2D[idxs[0]].astype('float32'), name='trans2D', borrow=True)
        tsJL = theano.shared(self.jointLength[0].astype('float32'), name='joint length', borrow=True)
        tsJB = theano.shared(self.jointBounds[0].astype('float32'), name='joint bound', borrow=True)
        tsPB = theano.shared(numpy.ones((max([len(i) for i in self.li_posebitIdx]),)).astype('float32'), name='posebit', borrow=True)
        tsPBS = theano.shared(numpy.zeros((self.numJoints*3, max([len(i) for i in self.li_posebitIdx]))).astype('float32'), name='posebitS', borrow=True)

        # 3D -> 2D projection also shift by M to cropped window
        Li_subset = tsLi
        Li_subset_glob3D = (T.reshape(Li_subset, (self.numJoints, 3))*tsScale+tsOff3D.dimshuffle('x', 0)).reshape((self.numJoints, 3))
        Li_subset_glob3D_hom = T.concatenate([Li_subset_glob3D, T.ones((self.numJoints, 1), dtype='float32')], axis=1)
        Li_subset_glob2D_hom = T.dot(Li_subset_glob3D_hom, tsCam.T)
        Li_subset_glob2D_hom = T.set_subtensor(Li_subset_glob2D_hom[:, 0], T.true_div(Li_subset_glob2D_hom[:, 0], Li_subset_glob2D_hom[:, 3]))
        Li_subset_glob2D_hom = T.set_subtensor(Li_subset_glob2D_hom[:, 1], T.true_div(Li_subset_glob2D_hom[:, 1], Li_subset_glob2D_hom[:, 3]))
        Li_subset_glob2D = T.set_subtensor(Li_subset_glob2D_hom[:, 2], T.true_div(Li_subset_glob2D_hom[:, 2], Li_subset_glob2D_hom[:, 3]))
        Li_subset_glob2D = Li_subset_glob2D[:, 0:3].reshape((self.numJoints, 3))
        Li_subset_img2D_hom = T.dot(Li_subset_glob2D, tsTrans2D)
        Li_subset_img2D_hom=T.set_subtensor(Li_subset_img2D_hom[:, 0], T.true_div(Li_subset_img2D_hom[:, 0], Li_subset_img2D_hom[:, 2]))
        Li_subset_img2D_hom=T.set_subtensor(Li_subset_img2D_hom[:, 1], T.true_div(Li_subset_img2D_hom[:, 1], Li_subset_img2D_hom[:, 2]))
        Li_subset_img2D = T.set_subtensor(Li_subset_img2D_hom[:, 2], T.true_div(Li_subset_img2D_hom[:, 2], Li_subset_img2D_hom[:, 2]))
        Li_subset_img2D = Li_subset_img2D[:, 0:2].reshape((self.numJoints*2,))
        Li_subset_img2Dcrop = (Li_subset_img2D - (self.Di.shape[3]/2.)) / (self.Di.shape[3]/2.)
        costP = self.ref_lambdaP/self.numJoints*T.sum(T.sqr(Li_subset_img2Dcrop - tsli))  # 3D to 2D projection

        # constraint with known joint length
        if len(self.lenghtConstraintIdx) > 0:
            di = T.dot(T.sqr(T.dot(tsLi, self.lenghtConstraint)), self.lenghtConstraintS)
            eq = (di - T.sqr(tsJL)).flatten()
        else:
            eq = theano.shared(numpy.cast['float32'](0.), name='dummy', borrow=True)

        costP += tsMuLagr/len(self.lenghtConstraintIdx)*T.sum(T.sqr(eq))

        # constraint with bounded joint length
        if len(self.boundConstraintIdx) > 0:
            di = T.dot(T.sqr(T.dot(tsLi, self.boundConstraint)), self.boundConstraintS)
            ieq = T.concatenate([(di - T.sqr(tsJB[:, 0])).flatten(), (-di + T.sqr(tsJB[:, 1])).flatten()], axis=0)
        else:
            ieq = None

        # posebits
        if max([len(i) for i in self.li_posebitIdx]) > 0:
            di = T.dot(tsLi, tsPBS)*tsPB - self.pb_thresh/tsScale
            if ieq is None:
                ieq = di.flatten()
            else:
                ieq = T.concatenate([ieq, di.flatten()], axis=0)

        # joint occupacy
        num_occupacy = 0
        if jointOccupacy > 0:
            num_occupacy = len(list(itertools.combinations(numpy.arange(self.numJoints), 2)))
            pairM = numpy.zeros((3*self.numJoints, 3*num_occupacy), dtype='float32')
            ip = 0
            for p in itertools.combinations(numpy.arange(self.numJoints), 2):
                # x coordinate
                pairM[3 * p[0] + 0, ip] = 1
                pairM[3 * p[1] + 0, ip] = -1
                # y coordinate
                ip += 1
                pairM[3 * p[0] + 1, ip] = 1
                pairM[3 * p[1] + 1, ip] = -1
                # z coordinate
                ip += 1
                pairM[3 * p[0] + 2, ip] = 1
                pairM[3 * p[1] + 2, ip] = -1
                ip += 1
            pairS = numpy.zeros((pairM.shape[1], pairM.shape[1]//3), dtype='float32')
            for ip in range(pairM.shape[1]//3):
                pairS[3*ip:3*(ip+1), ip] = 1

            di = T.dot(T.sqr(T.dot(tsLi, pairM)), pairS)
            oerr = (di - T.min(T.stack([T.sqr(jointOccupacy/tsScale), T.sqr(tsJL.min()-3.)])))
            if ieq is None:
                ieq = oerr.flatten()
            else:
                ieq = T.concatenate([ieq, oerr.flatten()], axis=0)

        # zig-zag constraint
        if True:
            v1 = T.dot(tsLi, self.zz_pairs_v1M)
            v2 = T.dot(tsLi, self.zz_pairs_v2M)
            vp = v1*v2
            di = T.dot(vp, self.zz_pairS)
            zzerr = (di-self.zz_thresh)
            if ieq is None:
                ieq = zzerr.flatten()
            else:
                ieq = T.concatenate([ieq, zzerr.flatten()], axis=0)

        print "compiling functions..."
        givens = []
        fun_cost = theano.function([], costP, givens=givens, mode='FAST_RUN', on_unused_input='warn')

        grad_cost = T.grad(costP, tsLi)
        gf_cost = theano.function([], grad_cost, givens=givens, mode='FAST_RUN', on_unused_input='warn')

        fun_eq_constr = theano.function([], eq, givens=givens, mode='FAST_RUN', on_unused_input='warn')

        idx = T.lscalar('idx')
        grad_eq_constr = T.grad(eq[idx], tsLi)
        gf_eq_constr = theano.function([idx], grad_eq_constr, givens=givens, mode='FAST_RUN', on_unused_input='warn')

        fun_ieq_constr = theano.function([], ieq, givens=givens, mode='FAST_RUN', on_unused_input='warn')

        grad_ieq_constr = T.grad(ieq[idx], tsLi)
        gf_ieq_constr = theano.function([idx], grad_ieq_constr, givens=givens, mode='FAST_RUN', on_unused_input='warn')
        print "done"

        # creates a function that computes the cost
        def train_fn(x, ref_tsLi):
            ref_tsLi.set_value(x.reshape(li3D[0].shape).astype('float32'))
            return numpy.asarray(fun_cost()).flatten().astype(float)

        # creates a function that computes the gradient
        def train_fn_grad(x, ref_tsLi):
            ref_tsLi.set_value(x.reshape(li3D[0].shape).astype('float32'))
            return numpy.asarray(gf_cost()).flatten().astype(float)

        # creates a function that computes the equality cost
        def eq_constr_fn(x, ref_tsLi):
            ref_tsLi.set_value(x.reshape(li3D[0].shape).astype('float32'))
            return numpy.asarray(fun_eq_constr()).flatten().astype(float)

        # creates a function that computes the gradient of equality cost
        def eq_constr_fn_grad(x, ref_tsLi):
            arr = numpy.zeros((len(self.lenghtConstraintIdx), x.size))
            ref_tsLi.set_value(x.reshape(li3D[0].shape).astype('float32'))
            for ii in range(len(self.lenghtConstraintIdx)):
                arr[ii] = numpy.asarray(gf_eq_constr(ii)).flatten()
            return arr.astype(float)

        # creates a function that computes the equality cost
        def ieq_constr_fn(x, ref_tsLi):
            ref_tsLi.set_value(x.reshape(li3D[0].shape).astype('float32'))
            # print numpy.asarray(fun_ieq_constr())
            return numpy.asarray(fun_ieq_constr()).flatten().astype(float)

        # creates a function that computes the gradient of equality cost
        def ieq_constr_fn_grad(x, ref_tsLi):
            ieqsz = fun_ieq_constr().size
            arr = numpy.zeros((ieqsz, x.size))
            ref_tsLi.set_value(x.reshape(li3D[0].shape).astype('float32'))
            for ii in range(ieqsz):
                arr[ii] = numpy.asarray(gf_ieq_constr(ii)).flatten()
            return arr.astype(float)

        # SLSQP can only take small number of unknown separately, so we iterate over each sample
        pbar = pb.ProgressBar(maxval=len(idxs), widgets=['Running SLSQP', pb.Percentage(), pb.Bar()])
        pbar.start()

        start = time.time()
        for i in xrange(len(idxs)):

            li_idx = numpy.where(self.subset_idxs == idxs[i])[0][0]

            # switch shared data
            tsLi.set_value(li3D[i].astype('float32'), borrow=True)
            tsli.set_value(self.li[li_idx].astype('float32'), borrow=True)
            tsScale.set_value(self.Di_scale[idxs[i]].astype('float32'), borrow=True)
            tsOff3D.set_value(self.Di_off3D[idxs[i]].astype('float32'), borrow=True)
            tsTrans2D.set_value(self.Di_trans2D[idxs[i]].astype('float32'), borrow=True)
            tsJL.set_value(self.jointLength[self.hcIdxForFlatIdx(idxs[i])].astype('float32'), borrow=True)
            tsJB.set_value(self.jointBounds[self.hcIdxForFlatIdx(idxs[i])].astype('float32'), borrow=True)
            pb_val = numpy.ones((len(self.li_posebitIdx[li_idx]),)).astype('float32')
            for p in range(len(self.li_posebitIdx[li_idx])):
                if self.li_posebits[li_idx][p] != self.li_posebitIdx[li_idx][p]:
                    pb_val[p] *= (-1.)
            tsPB.set_value(pb_val, borrow=True)

            posebitS = numpy.zeros((self.numJoints*3, len(self.li_posebitIdx[li_idx])), dtype='float32')
            for ip in range(len(self.li_posebitIdx[li_idx])):
                posebitS[self.li_posebitIdx[li_idx][ip][0]*3+2, ip] = -1.
                posebitS[self.li_posebitIdx[li_idx][ip][1]*3+2, ip] = 1.
            tsPBS.set_value(posebitS, borrow=True)

            # check bounds with GT
            if self.gt3D is not None:
                bnds = self.getReferenceBounds(self.gt3D[idxs[i]], idxs[i])
                for j in xrange(self.gt3D[idxs[i]].flatten().shape[0]):
                    if not (bnds[j][0] < self.gt3D[idxs[i]].flatten()[j] < bnds[j][1]):
                        print "Bound violated: Sample ", i, " joint ", j, bnds[j][0], "<", self.gt3D[idxs[i]].flatten()[j], "<", bnds[j][1]

            bnds = self.getReferenceBounds(li3D[i], idxs[i])

            x0 = li3D[i].flatten()
            # project on feasible set
            bounds = numpy.asarray(bnds)
            x0 = numpy.median(numpy.column_stack([bounds[:, 0], x0, bounds[:, 1]]), axis=1)

            if 'ref_optimize_incHC' in self.eval_params:
                try:
                    if self.eval_params['ref_optimize_incHC'] is True:
                        best_rval = [x0]
                        tsMuLagr.set_value(numpy.cast['float32'](self.ref_muLagr))
                        for k in range(5):
                            best_rval = scipy.optimize.fmin_slsqp(
                                func=train_fn,
                                x0=best_rval[0],
                                fprime=train_fn_grad,
                                # f_eqcons=eq_constr_fn,
                                # fprime_eqcons=eq_constr_fn_grad,
                                f_ieqcons=ieq_constr_fn,
                                fprime_ieqcons=ieq_constr_fn_grad,
                                bounds=bnds,
                                disp=0,
                                full_output=True,
                                iter=100,
                                acc=1e-5,
                                args=(tsLi,)
                            )
                            tsMuLagr.set_value(numpy.cast['float32'](tsMuLagr.get_value()*10.))
                    else:
                        best_rval = scipy.optimize.fmin_slsqp(
                            func=train_fn,
                            x0=x0,
                            fprime=train_fn_grad,
                            # f_eqcons=eq_constr_fn,
                            # fprime_eqcons=eq_constr_fn_grad,
                            f_ieqcons=ieq_constr_fn,
                            fprime_ieqcons=ieq_constr_fn_grad,
                            bounds=bnds,
                            disp=0,
                            full_output=True,
                            iter=100,
                            acc=1e-5,
                            args=(tsLi,)
                        )
                except:
                    print "Error while optimizing sample {}".format(idxs[i])
                    raise
            else:
                raise NotImplementedError("Missing parameter 'ref_optimize_incHC'")

            li3D[i] = best_rval[0].reshape(li3D[0].shape)

            pbar.update(i)

            # show initialization
            # orig2D = self.project3Dto2D(orig[i], idxs[i])*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)
            # li2D = self.project3Dto2D(li3D[i], idxs[i])*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)
            # if self.normZeroOne is True:
            #     dpt = self.Di[idxs[i], 0]*(2.*self.Di_scale[idxs[i]]) + (self.Di_off3D[idxs[i], 2] - self.Di_scale[idxs[i]])
            # else:
            #     dpt = self.Di[idxs[i], 0]*self.Di_scale[idxs[i]] + self.Di_off3D[idxs[i], 2]
            # dpt[dpt == (self.Di_off3D[idxs[i], 2] + self.Di_scale[idxs[i]])] = 0.
            # self.hpe.plotResult(dpt, orig2D.reshape((self.numJoints, 2)), li2D.reshape((self.numJoints, 2)), niceColors=True, visibility=self.li_visiblemask[i])
            # self.hpe.plotResult3D(dpt, self.Di_trans2D[idxs[i]].T,
            #                  orig[i]*self.Di_scale[idxs[i]]+self.Di_off3D[idxs[i]],
            #                  orig[i]*self.Di_scale[idxs[i]]+self.Di_off3D[idxs[i]],
            #                  niceColors=True, visibility=self.li_visiblemask[i])
            # if self.gt3D is not None:
            #     self.hpe.plotResult3D(dpt, self.Di_trans2D[idxs[i]].T,
            #                      self.gt3D[idxs[i]].reshape((self.numJoints, 3))*self.Di_scale[idxs[i]]+self.Di_off3D[idxs[i]],
            #                      self.gt3D[idxs[i]].reshape((self.numJoints, 3))*self.Di_scale[idxs[i]]+self.Di_off3D[idxs[i]],
            #                      niceColors=True, visibility=self.li_visiblemask[i])
            # self.hpe.plotResult3D(dpt, self.Di_trans2D[idxs[i]].T,
            #                  li3D[i].reshape((self.numJoints, 3))*self.Di_scale[idxs[i]]+self.Di_off3D[idxs[i]],
            #                  li3D[i].reshape((self.numJoints, 3))*self.Di_scale[idxs[i]]+self.Di_off3D[idxs[i]],
            #                  niceColors=True, visibility=self.li_visiblemask[i])

        pbar.finish()
        print "Took {}s for reference frame optimization".format(time.time()-start)

        # Clean up, to free possible gpu memory for several function calls
        del tsli, tsMuLagr, tsCam, tsScale, tsOff3D, tsTrans2D, tsJL, tsJB, tsPB, tsPBS, tsLi
        gc.collect()
        gc.collect()
        gc.collect()

        return li3D

    def selectReferenceFrames(self):

        if 'ref_descriptor' in self.eval_params:
            if self.eval_params['ref_descriptor'] == 'hog_msk':
                # the reference frames get normalized with mask, the rest for testing NOT
                # they only get normalized if there is NO mask involved at all
                if self.useCache:
                    all_emb = self.Ri
                    ref_emb, ref_mask = self.getImageDescriptors_HOG(self.Di)
                else:
                    all_emb, _ = self.getImageDescriptors_HOG(self.Di, useMask=False, doNormalize=False)
                    ref_emb, ref_mask = self.getImageDescriptors_HOG(self.Di)
                dist = self.getImageDescriptors_HOG_cdist(all_emb, ref_emb, ref_mask)
                dist -= dist.min()
                dist /= dist.max()
                dist *= (-1.)
                dist += 1.
            elif self.eval_params['ref_descriptor'] == 'hog':
                if self.useCache:
                    all_emb = self.Ri
                else:
                    all_emb, _ = self.getImageDescriptors_HOG(self.Di, useMask=False, doNormalize=True)
                dist = pairwise_distances(all_emb, metric='cosine')
                dist -= dist.min()
                dist /= dist.max()
            else:
                raise NotImplementedError("Unknown parameter: "+self.eval_params['ref_descriptor'])
        else:
            raise NotImplementedError("Missing parameter 'ref_descriptor'")

        import matplotlib.pyplot as plt
        if self.numSamples < self.detailEvalThresh:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(dist, interpolation='none')
            plt.show(block=True)
            fig.savefig(self.eval_prefix+'/ref_sim_{}.png'.format(self.eval_params['ref_descriptor']),
                        bbox_inches='tight')
            plt.close(fig)

        if (self.gt3D is not None) and (self.numSamples < self.detailEvalThresh):
            all_pose = self.gt3D.reshape(self.numSamples, -1)
            dist_p = pairwise_distances(all_pose, metric='euclidean')
            dist_p -= dist_p.min()
            dist_p /= dist_p.max()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            hist, xbins, ybins = numpy.histogram2d(dist.ravel(), dist_p.ravel(), bins=100)
            hist /= hist.sum(axis=0)[None, :]
            extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
            ax.imshow(hist.T, interpolation='none', origin='lower', extent=extent)
            ax.set_xlabel("output")
            ax.set_ylabel("|delta pose|^2")
            ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2])))
            plt.show(block=True)
            fig.savefig(self.eval_prefix+'/ref_scatter_{}.png'.format(self.eval_params['ref_descriptor']), bbox_inches='tight')
            plt.close(fig)

            # first nearest neighbor
            fig = plt.figure()
            ax = fig.add_subplot(111)
            hist, xbins, ybins = numpy.histogram2d(dist[numpy.arange(dist.shape[0]), dist.argsort(axis=1)[:, 1]].ravel(),
                      dist_p[numpy.arange(dist.shape[0]), dist.argsort(axis=1)[:, 1]].ravel(), bins=100)
            extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
            ax.imshow(hist.T, interpolation='none', origin='lower', extent=extent)
            ax.set_xlabel("output")
            ax.set_ylabel("|delta pose|^2")
            ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2])))
            plt.show(block=True)
            fig.savefig(self.eval_prefix+'/ref_nn_{}.png'.format(self.eval_params['ref_descriptor']), bbox_inches='tight')
            plt.close(fig)

        # apply submodular clustering on similarity matrix
        dist[numpy.isnan(dist)] = 0.
        dist[numpy.isinf(dist)] = 0.
        start = time.time()

        if 'ref_cluster' in self.eval_params:
            if self.eval_params['ref_cluster'] == 'sm_greedy':
                if 'ref_fraction' in self.eval_params:
                    Nmax = int(self.numSamples*self.eval_params['ref_fraction']/100.)
                else:
                    raise NotImplementedError("Missing parameter 'ref_fraction'")

                if 'ref_threshold' in self.eval_params:
                    thrsh = self.eval_params['ref_threshold']
                else:
                    raise NotImplementedError("Missing parameter 'ref_threshold'")
                refs = submodularClusterGreedy(dist, thrsh, Nmax=Nmax)
            else:
                raise NotImplementedError("Unknown parameter: "+self.eval_params['ref_cluster'])
        else:
            raise NotImplementedError("Missing parameter 'ref_cluster'")

        print "Took {}s for clustering".format(time.time()-start)

        del dist
        gc.collect()
        gc.collect()
        gc.collect()

        return refs

    def evalReferenceFrameSelection(self):
        closest_joints = numpy.zeros((self.numSamples, self.numJoints, 3))
        # init with reference frames and skip
        if self.gt3D is not None:
            init = self.gt3D.reshape((self.numSamples, self.numJoints, 3))
        else:
            init = numpy.zeros((self.numSamples, self.numJoints, 3), dtype='float32')

        ref_idx = [0]*self.numSamples
        if (self.numSamples < self.detailEvalThresh) or (self.gt3D is not None):
            for i in xrange(self.numSamples):
                ref_idx[i], _, _ = self.getReferenceForSample(i, init, False)
                if self.gt3D is not None:
                    closest_joints[i] = self.gt3D[ref_idx[i]].reshape((self.numJoints, 3))*self.Di_scale[ref_idx[i]]

        if (self.debugPrint is True) and (self.numSamples < self.detailEvalThresh):
            sel = numpy.zeros((self.numSamples, self.numSubset), dtype='uint8')
            for i in xrange(self.numSamples):
                sel[i, numpy.where(self.subset_idxs == ref_idx[i])[0][0]] = 255

            cv2.imwrite("{}/ref_descriptor_{}_{}_{}.png".format(self.eval_prefix,
                                                                self.eval_params['ref_descriptor'],
                                                                self.eval_params['ref_cluster'],
                                                                self.eval_params['ref_threshold']), sel)

        # plot ROC for selection
        if self.gt3D is not None:
            numpy.save("{}/ref_sel_joints_{}_{}_{}.npy".format(self.eval_prefix,
                                                               self.eval_params['ref_descriptor'],
                                                               self.eval_params['ref_cluster'],
                                                               self.eval_params['ref_threshold']), closest_joints)
            gt3D = numpy.asarray([self.gt3D[idx] * self.Di_scale[idx] for idx in xrange(self.gt3D.shape[0])]).reshape(
                (self.numSamples, self.numJoints, 3))
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot([(numpy.nanmax(numpy.sqrt(numpy.square(gt3D - closest_joints).sum(axis=2)), axis=1) <= j).sum() / float(
                self.numSamples) * 100. for j in xrange(0, 80)])
            plt.xlabel('Distance threshold / mm')
            plt.ylabel('Fraction of frames within distance / %')
            plt.ylim([0.0, 100.0])
            ax.grid(True)
            plt.show(block=False)
            fig.savefig("{}/ref_sel_ROC_{}_{}_{}.pdf".format(self.eval_prefix,
                                                             self.eval_params['ref_descriptor'],
                                                             self.eval_params['ref_cluster'],
                                                             self.eval_params['ref_threshold']),
                        bbox_inches='tight')
            plt.close(fig)

    def getReferenceForSample(self, i, init3D, doOffset=True):
        """
        Gets closest reference frame in appearance for given frame
        therefore shift sample frame by some offset to find best alignment
        :param i: sample index
        :param init3D: 3D locations of joint data
        :return: reference frame index, offset 2D in px, offset 3D in mm
        """

        unannotated = numpy.setdiff1d(numpy.arange(self.numSamples), self.subset_idxs, assume_unique=True)

        def idxToOff(boi):
            ad = boi % 5
            st = boi // 5 + 1
            if ad == 0:
                return numpy.asarray([[0, 0]]).repeat(self.numJoints, axis=0)
            elif ad == 1:
                return numpy.asarray([[int(shift*st), -int(shift*st)]]).repeat(self.numJoints, axis=0)
            elif ad == 2:
                return numpy.asarray([[-int(shift*st), int(shift*st)]]).repeat(self.numJoints, axis=0)
            elif ad == 3:
                return numpy.asarray([[int(shift*st), int(shift*st)]]).repeat(self.numJoints, axis=0)
            elif ad == 4:
                return numpy.asarray([[-int(shift*st), -int(shift*st)]]).repeat(self.numJoints, axis=0)
            else:
                raise NotImplementedError("Internal error!")

        shift = 2
        steps = 4

        if 'init_refwithinsequence' in self.eval_params:
            if self.eval_params['init_refwithinsequence'] is True:
                # sequence break is inclusive
                if i <= self.sequenceBreaks[0]:
                    seq_l = 0
                else:
                    d1 = i-numpy.asarray(self.sequenceBreaks)
                    d1[d1 <= 0] = self.numSamples+1
                    idx_k = numpy.argmin(d1)
                    seq_l = self.sequenceBreaks[idx_k]
                if i > self.sequenceBreaks[-1]:
                    seq_h = self.numSamples
                else:
                    d2 = i-numpy.asarray(self.sequenceBreaks)
                    d2[d2 > 0] = self.numSamples+1
                    idx_kp1 = numpy.argmin(numpy.abs(d2))
                    seq_h = self.sequenceBreaks[idx_kp1]
                mask = numpy.ones((self.numSubset,), dtype='bool')
                mask[numpy.bitwise_and(self.subset_idxs >= seq_l, self.subset_idxs <= seq_h)] = 0
                mask = numpy.tile(mask, (5*steps, 1))
            else:
                mask = numpy.zeros((5*steps, self.numSubset), dtype='bool')
        else:
            raise NotImplementedError("Missing parameter 'init_refwithinsequence'")

        # lazy init to save time
        if not hasattr(self, 'ref_emb'):
            if 'ref_descriptor' in self.eval_params:
                if self.eval_params['ref_descriptor'] == 'hog_msk':
                    if self.useCache:
                        self.ref_emb, self.ref_mask = self.getImageDescriptors_HOG(self.Di[self.subset_idxs, :, :, :])
                        self.sample_emb = self.Ri[unannotated]
                    else:
                        self.ref_emb, self.ref_mask = self.getImageDescriptors_HOG(self.Di[self.subset_idxs, :, :, :])
                        self.sample_emb, _ = self.getImageDescriptors_HOG(self.Di[unannotated, :, :, :],
                                                                          useMask=False, doNormalize=False)
                elif self.eval_params['ref_descriptor'] == 'hog':
                    if self.useCache:
                        self.ref_emb = self.Ri[self.subset_idxs]
                        self.sample_emb = self.Ri[unannotated]
                    else:
                        self.ref_emb, _ = self.getImageDescriptors_HOG(self.Di[self.subset_idxs, :, :, :],
                                                                       useMask=False, doNormalize=True)
                        self.sample_emb, _ = self.getImageDescriptors_HOG(self.Di[unannotated, :, :, :],
                                                                          useMask=False, doNormalize=True)
                else:
                    raise NotImplementedError("Unknown parameter: "+self.eval_params['ref_descriptor'])
            else:
                raise NotImplementedError("Missing parameter 'ref_descriptor'")

        queryImg = numpy.zeros((5*steps, 1, self.Di.shape[2], self.Di.shape[3]))
        for st in range(1, steps+1):
            for ad in range(5):
                # shift to 0, +1, -1
                if ad == 0:
                    off = (0, 0)
                elif ad == 1:
                    off = (int(shift*st), -int(shift*st))
                elif ad == 2:
                    off = (-int(shift*st), int(shift*st))
                elif ad == 3:
                    off = (int(shift*st), int(shift*st))
                elif ad == 4:
                    off = (-int(shift*st), -int(shift*st))

                imgD = self.Di[i, 0].copy()
                # shift by padding
                if off[0] > 0:
                    imgD = numpy.pad(imgD, ((0, 0), (off[0], 0)), mode='constant', constant_values=(1., ))[:, :-off[0]]
                elif off[0] < 0:
                    imgD = numpy.pad(imgD, ((0, 0), (0, -off[0])), mode='constant', constant_values=(1., ))[:, -off[0]:]

                if off[1] > 0:
                    imgD = numpy.pad(imgD, ((off[1], 0), (0, 0)), mode='constant', constant_values=(1., ))[:-off[1], :]
                elif off[1] < 0:
                    imgD = numpy.pad(imgD, ((0, -off[1]), (0, 0)), mode='constant', constant_values=(1., ))[-off[1]:, :]

                queryImg[(st-1)*5+ad, 0] = imgD

        if 'ref_descriptor' in self.eval_params:
            if self.eval_params['ref_descriptor'] == 'hog_msk':
                sample_emb, _ = self.getImageDescriptors_HOG(queryImg, useMask=False, doNormalize=False)
                dist = self.getImageDescriptors_HOG_cdist(sample_emb, self.ref_emb, self.ref_mask)
                best_off_idx, best_ref_idx = numpy.unravel_index(numpy.ma.array(dist, mask=mask).argmax(), dist.shape)
            elif self.eval_params['ref_descriptor'] == 'hog':
                sample_emb, _ = self.getImageDescriptors_HOG(queryImg, useMask=False, doNormalize=True)
                dist = pairwise_distances(sample_emb, self.ref_emb, metric='cosine')
                best_off_idx, best_ref_idx = numpy.unravel_index(numpy.ma.array(dist, mask=mask).argmin(), dist.shape)
            else:
                raise NotImplementedError("Unknown parameter: "+self.eval_params['ref_descriptor'])
        else:
            raise NotImplementedError("Missing parameter 'ref_descriptor'")

        # further alignment
        if 'init_offset' in self.eval_params:
            if doOffset:
                if self.eval_params['init_offset'] == 'siftflow':
                    # use offset in 2D wrt to the reference frame
                    if numpy.allclose(init3D[self.subset_idxs[best_ref_idx]], 0.):
                        print "WARNING: init is not set! No SIFTflow aligment possible."
                        warpImg = self.Di[i, 0]
                    else:
                        li2D = self.project3Dto2D(init3D[self.subset_idxs[best_ref_idx]],
                                                  self.subset_idxs[best_ref_idx]).reshape(self.numJoints, 2)
                        best_off2D_glob, warpImg = self.alignSIFTFlow(li2D, self.subset_idxs[best_ref_idx], i)
                    best_off3D_glob = numpy.zeros((self.numJoints, 3))
                else:
                    raise NotImplementedError("Unknown parameter 'init_offset' {}".format(self.eval_params['init_offset']))
            else:
                # reset offset
                best_off2D_glob = idxToOff(best_off_idx)
                best_off3D_glob = numpy.zeros((self.numJoints, 3))
                warpImg = self.Di[i, 0]
        else:
            raise NotImplementedError("Missing parameter 'init_offset'")

        # Check temporal samples
        if 'init_fallback' in self.eval_params:
            if self.eval_params['init_fallback'] is True:
                if i < numpy.min(self.subset_idxs):
                    idx_k = idx_kp1 = self.subset_idxs[0]
                elif i >= numpy.max(self.subset_idxs):
                    idx_k = idx_kp1 = self.subset_idxs[-1]
                else:
                    d1 = i-self.subset_idxs
                    d1[d1 < 0] = self.numSamples+1
                    idx_k = self.subset_idxs[numpy.argmin(d1)]
                    d2 = i-self.subset_idxs
                    d2[d2 >= 0] = self.numSamples+1
                    idx_kp1 = self.subset_idxs[numpy.argmin(numpy.abs(d2))]

                ssd_k = numpy.square(queryImg - self.Di[idx_k, 0][None, :, :]).sum(axis=2).sum(axis=1)
                ssd_kp1 = numpy.square(queryImg - self.Di[idx_kp1, 0][None, :, :]).sum(axis=2).sum(axis=1)
                ssd_emb = numpy.square(queryImg[best_off_idx] - warpImg).sum()

                am = numpy.argmin([ssd_emb, ssd_k.min(), ssd_kp1.min()])
                if am == 0:
                    best_ref_idx = best_ref_idx
                    best_off2D = best_off2D_glob
                    best_off3D = best_off3D_glob
                    print "Using flow"
                elif am == 1:
                    best_ref_idx = numpy.where(self.subset_idxs == idx_k)[0][0]
                    best_off2D = idxToOff(ssd_k.argmin())
                    best_off3D = numpy.zeros((self.numJoints, 3))
                    print "Using temp-1"
                elif am == 2:
                    best_ref_idx = numpy.where(self.subset_idxs == idx_kp1)[0][0]
                    best_off2D = idxToOff(ssd_kp1.argmin())
                    best_off3D = numpy.zeros((self.numJoints, 3))
                    print "Using temp+1"
                else:
                    raise NotImplementedError("Internal error")
            else:
                best_ref_idx = best_ref_idx
                best_off2D = best_off2D_glob
                best_off3D = best_off3D_glob
        else:
            raise NotImplementedError("Missing parameter 'init_fallback'")

        if self.debugPrint:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.imshow(numpy.concatenate([self.Di[i, 0], self.Di[self.subset_idxs[best_ref_idx], 0]]), cmap='gray')
            plt.show(block=False)
            fig.savefig(self.eval_prefix+'/pairs_{}.png'.format(i), bbox_inches='tight')
            plt.close(fig)
        print "BEST for i={}: ref={}, off2D={}, off3D={}".format(i, self.subset_idxs[best_ref_idx],
                                                             best_off2D.tolist(), best_off3D.tolist())

        return self.subset_idxs[best_ref_idx], best_off2D, best_off3D

    def getClosestSampleForReference(self, init3D):
        """
        Given the reference frames, get the closest unannotated sample in appearance
        :param init3D: 3D location of joint data
        :return: sample frame index, reference frame index
        """

        unannotated = numpy.setdiff1d(numpy.arange(self.numSamples), self.subset_idxs, assume_unique=True)

        # lazy init to save time
        if not hasattr(self, 'ref_emb'):
            if 'ref_descriptor' in self.eval_params:
                if self.eval_params['ref_descriptor'] == 'hog_msk':
                    if self.useCache:
                        self.ref_emb, self.ref_mask = self.getImageDescriptors_HOG(self.Di[self.subset_idxs, :, :, :])
                        self.sample_emb = self.Ri[unannotated]
                    else:
                        self.ref_emb, self.ref_mask = self.getImageDescriptors_HOG(self.Di[self.subset_idxs, :, :, :])
                        self.sample_emb, _ = self.getImageDescriptors_HOG(self.Di[unannotated, :, :, :],
                                                                          useMask=False, doNormalize=False)
                elif self.eval_params['ref_descriptor'] == 'hog':
                    if self.useCache:
                        self.ref_emb = self.Ri[self.subset_idxs]
                        self.sample_emb = self.Ri[unannotated]
                    else:
                        self.ref_emb, _ = self.getImageDescriptors_HOG(self.Di[self.subset_idxs, :, :, :],
                                                                       useMask=False, doNormalize=True)
                        self.sample_emb, _ = self.getImageDescriptors_HOG(self.Di[unannotated, :, :, :],
                                                                          useMask=False, doNormalize=True)
                else:
                    raise NotImplementedError("Unknown parameter: "+self.eval_params['ref_descriptor'])
            else:
                raise NotImplementedError("Missing parameter 'ref_descriptor'")

        # check consistency
        assert self.sample_emb.shape[0] + self.ref_emb.shape[0] == self.numSamples
        assert self.ref_emb.shape[0] == len(self.subset_idxs)

        if 'ref_descriptor' in self.eval_params:
            if self.eval_params['ref_descriptor'] == 'hog_msk':
                dist = self.getImageDescriptors_HOG_cdist(self.sample_emb, self.ref_emb, self.ref_mask)
                best_sample_idx, best_ref_idx = numpy.unravel_index(numpy.argmax(dist), dist.shape)
            elif self.eval_params['ref_descriptor'] == 'hog':
                dist = pairwise_distances(self.sample_emb, self.ref_emb, metric='cosine')
                best_sample_idx, best_ref_idx = numpy.unravel_index(numpy.argmin(dist), dist.shape)
            else:
                raise NotImplementedError("Unknown parameter: "+self.eval_params['ref_descriptor'])
        else:
            raise NotImplementedError("Missing parameter 'ref_descriptor'")

        # further alignment
        if 'init_offset' in self.eval_params:
            if self.eval_params['init_offset'] == 'siftflow':
                # use offset in 2D wrt to the reference frame
                if numpy.allclose(init3D[self.subset_idxs[best_ref_idx]], 0.):
                    print "WARNING: init is not set! No SIFTflow aligment possible."
                else:
                    li2D = self.project3Dto2D(init3D[self.subset_idxs[best_ref_idx]],
                                              self.subset_idxs[best_ref_idx]).reshape(self.numJoints, 2)
                    off2D, _ = self.alignSIFTFlow(li2D, self.subset_idxs[best_ref_idx], unannotated[best_sample_idx])
                off3D = numpy.zeros((self.numJoints, 3))
            else:
                raise NotImplementedError("Unknown parameter 'init_offset' {}".format(self.eval_params['init_offset']))
        else:
            raise NotImplementedError("Missing parameter 'init_offset'")

        if self.debugPrint:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.imshow(numpy.concatenate([self.Di[unannotated[best_sample_idx], 0],
                                          self.Di[self.subset_idxs[best_ref_idx], 0]]), cmap='gray')
            plt.show(block=False)
            fig.savefig(self.eval_prefix+'/pairs_{}.png'.format(unannotated[best_sample_idx]), bbox_inches='tight')
            plt.close(fig)

        print "BEST for i={}: ref={}, off2D={}, off3D={}".format(unannotated[best_sample_idx],
                                                                 self.subset_idxs[best_ref_idx],
                                                                 off2D.tolist(), off3D.tolist())

        return unannotated[best_sample_idx], self.subset_idxs[best_ref_idx], off2D, off3D

    def alignSIFTFlow(self, li2D, ref_idx, idx):
        """
        Align two images with SIFT flow and shift annotations according to flow
        :param li2D: original annotations, normalized 2D
        :param ref_idx: reference frame index
        :param idx: query frame index
        :return: shifted annotations
        """

        start = time.time()
        if False:
            raise NotImplementedError("!")
        else:
            # quick and dirty call of matlab script
            scipy.io.savemat("./etc/sift_flow/input_{}.mat".format(os.getpid()),
                             {'in1': self.Di[ref_idx, 0], 'in2': self.Di[idx, 0]})
            cmd = "-nojvm -nodisplay -nosplash -r \"cd ./etc/sift_flow/; procid={}; try, run('process.m'); end; quit \" ".format(os.getpid())

            from subprocess import call
            call(["matlab", cmd])
            data = scipy.io.loadmat("./etc/sift_flow/output_{}.mat".format(os.getpid()))
            flow = data['flow'].astype('float')
            vx = flow[:, :, 0]
            vy = flow[:, :, 1]
            warpI2 = data['warpI2']

        # find closest offset vector from siftflow an shift 2D annoataion by this amount
        off = numpy.zeros((self.numJoints, 2))
        for j in xrange(self.numJoints):
            x = numpy.rint(li2D[j, 0]*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)).astype(int)
            y = numpy.rint(li2D[j, 1]*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)).astype(int)
            # only valid pixels, discard offsets out of image
            if (0 <= x <= self.Di.shape[3]-1) and (0 <= y <= self.Di.shape[2]-1):
                off[j, 0] = -vx[y, x]
                off[j, 1] = -vy[y, x]

        print "Took {}s for align SIFTFlow".format(time.time()-start)

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(vx)
        # plt.show(block=True)
        # plt.close(fig)

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(warpI2)
        # plt.show(block=True)
        # plt.close(fig)

        return off, warpI2

    def getImageDescriptors_HOG_cdist(self, all_emb, ref_emb, ref_mask):
        # unnormalized cosine distance for HOG
        dist = numpy.dot(all_emb, ref_emb.T)
        # normalize by length of query descriptor projected on reference
        norm = numpy.sqrt(numpy.dot(numpy.square(all_emb), ref_mask.T))
        dist /= norm
        dist[numpy.isinf(dist)] = 0.
        dist[numpy.isnan(dist)] = 0.

        # dist[numpy.triu_indices(dist.shape[0], 1)] = numpy.maximum(dist[numpy.triu_indices(dist.shape[0], 1)],
        #                                                            dist.T[numpy.triu_indices(dist.shape[0], 1)])
        # dist[numpy.tril_indices(dist.shape[0], -1)] = 0.
        # dist += dist.T

        return dist

    def getImageDescriptors_HOG(self, images, useMask=True, doNormalize=True):
        win_size = (128, 128)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        start = time.time()

        def createBlockMask(msk):
            blockMask = numpy.zeros((win_size[1]//cell_size[1], win_size[0]//cell_size[0]), dtype='bool')
            for y in range(win_size[1]//cell_size[1]):
                for x in range(win_size[0]//cell_size[0]):
                    # 1/4 of pixels in cell must be set
                    if numpy.sum(msk[cell_size[1]*y:cell_size[1]*(y+1), cell_size[0]*x:cell_size[0]*(x+1)]) > cell_size[0]*cell_size[1]//4:
                        blockMask[y, x] = True
            return blockMask

        def maskDescr(descriptors, msk, blockMask):
            blocks_in_x_dir = win_size[0]//block_stride[0] - 1
            blocks_in_y_dir = win_size[1]//block_stride[1] - 1
            dataCellStart = 0
            for blockx in range(blocks_in_x_dir):
                for blocky in range(blocks_in_y_dir):
                    # cells per block ...
                    for cellNr in range((block_size[0]//cell_size[0])*(block_size[1]//cell_size[1])):
                        # compute corresponding cell nr
                        cellx = blockx
                        celly = blocky
                        celly += cellNr % (block_size[0]//cell_size[0])
                        cellx += cellNr // (block_size[0]//cell_size[0])

                        if not blockMask[celly, cellx]:
                            descriptors[dataCellStart:dataCellStart+nbins] = 0.  # -1.0 set it to zero, no need to handle
                            msk[dataCellStart:dataCellStart+nbins] = 0.

                        dataCellStart += nbins

            return descriptors, msk

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        descr = numpy.zeros((images.shape[0], hog.getDescriptorSize()))
        mask = numpy.ones_like(descr)

        for i in xrange(images.shape[0]):
            img = images[i, 0].copy()
            if self.normZeroOne:
                img *= (2.*self.Di_scale[i])
            else:
                img *= self.Di_scale[i]
                img += self.Di_scale[i]
            img = cv2.normalize(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            descr[i] = hog.compute(img.astype('uint8'))[:, 0]

            if useMask:
                mk = numpy.bitwise_and(images[i, 0] < 1.-1e-4, images[i, 0] > -1.+1e-4)
                blockMask = createBlockMask(mk)
                descr[i], mask[i] = maskDescr(descr[i], mask[i], blockMask)

            if doNormalize:
                # normalize to unit length
                denom = numpy.sqrt(numpy.square(descr[i]).sum())
                if numpy.allclose(denom, 0.):
                    print "WARNING: Image {} has empty descriptor!".format(i)
                    denom = 1.
                descr[i] /= denom

        print "Took {}s for HOG embedding".format(time.time()-start)

        return descr, mask

    def getReferenceBounds(self, li3D, idx):
        """
        Get bounds for joints depth for reference frames
        :param li3D: 3D joints for reference frames
        :param idx: frame index
        :return: list with bounds
        """

        bnds = []
        li3D = li3D.reshape((self.numJoints, 3))
        x2D = self.project3Dto2D(li3D.reshape(self.numJoints, 3), idx) * (self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)
        x2D = x2D.reshape((self.numJoints, 2))
        # project from normalized 2D to global 2D
        li_img2D_hom = numpy.concatenate([x2D.reshape((1, self.numJoints, 2)),
                                          numpy.ones((1, self.numJoints, 1))], axis=2)
        invM = numpy.linalg.inv(self.Di_trans2D[idx]).reshape((1, 3, 3))
        li_glob2D = numpy.einsum('ijk,ikl->ijl', li_img2D_hom, invM)
        li_glob2D = li_glob2D[:, :, 0:2] / li_glob2D[:, :, 2][:, :, None]
        dm = self.importer.loadDepthMap(self.depthFiles[idx])
        for k in xrange(self.numJoints):
            iy = numpy.rint(li_glob2D[0, k, 1]).astype(int).clip(0, dm.shape[0]-1)
            ix = numpy.rint(li_glob2D[0, k, 0]).astype(int).clip(0, dm.shape[1]-1)
            if False:
                # median within 3x3 mask more robust to noise
                dpt = dm[iy-1:iy+2, ix-1:ix+2]
                dpt = numpy.median(dpt[numpy.bitwise_and(dpt >= (self.Di_off3D[idx, 2] - self.Di_scale[idx]),
                                                         dpt <= (self.Di_off3D[idx, 2] + self.Di_scale[idx]))])
                # no valid dpt sample in mask or directly over background, deal with this later
                if numpy.isnan(dpt) or numpy.isclose(dm[iy, ix], self.importer.getDepthMapNV()):
                    dpt = (self.Di_off3D[idx, 2] + self.Di_scale[idx])
            else:
                if not numpy.allclose(dm[iy, ix], self.importer.getDepthMapNV()) and (
                    self.Di_off3D[idx, 2] - self.Di_scale[idx]) <= dm[iy, ix] <= (
                            self.Di_off3D[idx, 2] + self.Di_scale[idx]):
                    dpt = dm[iy, ix]
                else:
                    dpt = (self.Di_off3D[idx, 2] + self.Di_scale[idx])

            bnds.append((-1., 1.))
            bnds.append((-1., 1.))
            if idx in self.subset_idxs and numpy.allclose(self.li_visiblemask[numpy.where(self.subset_idxs == idx)[0][0], k], 1.):
                if not numpy.allclose(dpt, (self.Di_off3D[idx, 2] + self.Di_scale[idx])):
                    if k in self.tips:
                        zm = (dpt - self.Di_off3D[idx, 2]) / self.Di_scale[idx]
                    else:
                        if isinstance(self.jointOff, (int, float)):
                            zm = (dpt + self.jointOff - self.Di_off3D[idx, 2]) / self.Di_scale[idx]
                        else:
                            zm = (dpt + self.jointOff[k][0] - self.Di_off3D[idx, 2]) / self.Di_scale[idx]
                    # zp = (dpt + self.jointEps - self.Di_off3D[idx, 2]) / self.Di_scale[idx]
                    zp = zm + self.jointEps/self.Di_scale[idx]
                    bnds.append((zm, zp))
                else:
                    # zm = li3D[k, 2]-self.jointEps/self.Di_scale[idx]
                    # zp = li3D[k, 2]+self.jointEps/self.Di_scale[idx]
                    # bnds.append((zm, zp))
                    bnds.append((-1., 1.))
            else:
                if not numpy.allclose(dpt, (self.Di_off3D[idx, 2] + self.Di_scale[idx])):
                    if k in self.tips:
                        zm = (dpt - self.Di_off3D[idx, 2]) / self.Di_scale[idx]
                    else:
                        if isinstance(self.jointOff, (int, float)):
                            zm = (dpt + self.jointOff - self.Di_off3D[idx, 2]) / self.Di_scale[idx]
                        else:
                            zm = (dpt + self.jointOff[k][0] - self.Di_off3D[idx, 2]) / self.Di_scale[idx]
                    bnds.append((zm, 1.))
                else:
                    bnds.append((-1., 1.))

        return bnds

    def evaluateToGT(self, Li, idxs):
        """
        Evaluate the current estimate to a ground truth
        :param Li: current estimates
        :param idxs: idxs to evaluate
        :return: mean error, max error and MD score
        """

        if not isinstance(idxs, numpy.ndarray):
            idxs = numpy.asarray(idxs)

        if self.gt3D is not None:
            gt3D_subset = self.gt3D[idxs]
            if Li.shape[0] == len(idxs):
                Li_subset = Li
            else:
                Li_subset = Li[idxs]
            mean_error = numpy.mean(numpy.sqrt(numpy.square((gt3D_subset - Li_subset.reshape(gt3D_subset.shape))*self.Di_scale[idxs, None, None]).sum(axis=2)), axis=1).mean()
            max_error = numpy.max(numpy.sqrt(numpy.square((gt3D_subset - Li_subset.reshape(gt3D_subset.shape))*self.Di_scale[idxs, None, None]).sum(axis=2)))
            vals = [(numpy.nanmax(numpy.sqrt(numpy.square((gt3D_subset - Li_subset.reshape(gt3D_subset.shape))*self.Di_scale[idxs, None, None]).sum(axis=2)), axis=1) <= j).sum() / float(gt3D_subset.shape[0]) for j in range(0, 80)]
            md_score = numpy.asarray(vals).sum() / float(80.)

            return mean_error, max_error, md_score
        else:
            return 0., 0., 0.

    def initClosestReference3D_maps(self, li):
        """
        Interpolate the given 3D positions of li by propagating their positions to fit Li in size
        checks for sequence breaks and do not interpolate over them
        :return: interpolated 3D positions
        """

        import matplotlib.pyplot as plt
        import theano
        import theano.tensor as T

        corrMaps_k = numpy.zeros((1, self.numJoints, self.Di.shape[2], self.Di.shape[3]))
        init = numpy.zeros((self.numSamples, self.numJoints, 3), dtype='float32')
        init_closest = numpy.zeros((self.numSamples, self.numJoints+1, 3), dtype='float32')  # +1 for reference idx
        init_aligned_cache = False
        if self.useCache and os.path.isfile("{}/Li_init_{}_aligned_{}.npy".format(self.eval_prefix,
                                                                                  self.eval_params['init_method'],
                                                                                  self.eval_params['init_offset'])):
            init_aligned = numpy.load("{}/Li_init_{}_aligned_{}.npy".format(self.eval_prefix,
                                                                            self.eval_params['init_method'],
                                                                            self.eval_params['init_offset']))
            init_aligned_cache = True
            if not numpy.all(numpy.in1d(init_aligned[:, 0, :].ravel(), self.subset_idxs.ravel())):
                init_aligned = numpy.zeros((self.numSamples, self.numJoints+1, 3), dtype='float32')  # +1 for reference idx
                init_aligned_cache = False
                print "WARNING: Cached subset idxs do not match!"
        else:
            init_aligned = numpy.zeros((self.numSamples, self.numJoints+1, 3), dtype='float32')  # +1 for reference idx
            init_aligned_cache = False
        # init with reference frames and skip
        for it in self.subset_idxs:
            init[it] = self.li3D_aug[numpy.where(self.subset_idxs == it)[0][0]].reshape(init[0].shape)

        lambdaLagr = self.rng.randn(self.numLengthConstraints) * 0.0
        tsMuLagr = theano.shared(numpy.cast['float32'](self.init_muLagr), name='muLagr', borrow=True)

        #######################
        # create correlation maps
        pbar = pb.ProgressBar(maxval=self.numSamples, widgets=["Optimizing", pb.Percentage(), pb.Bar()])
        pbar.start()

        corrected = []
        start = time.time()

        for isa in xrange(self.numSamples):

            if init_aligned_cache is False:
                if 'init_incrementalref' in self.eval_params:
                    if self.eval_params['init_incrementalref'] is True:
                        unannotated = numpy.setdiff1d(numpy.arange(self.numSamples), self.subset_idxs, assume_unique=True)
                        print "Current subset: {}".format(numpy.sort(self.subset_idxs))
                        print "Unannotated frames: {}".format(len(unannotated))
                        if len(unannotated) == 0:
                            continue
                        i, ref_idx, off2D, off3D = self.getClosestSampleForReference(init)
                        assert i not in self.subset_idxs
                    else:
                        ref_idx, off2D, off3D = self.getReferenceForSample(isa, init)
                        i = isa
                else:
                    raise NotImplementedError("Missing parameter 'init_incrementalref'")

                # project init pose into current frame
                # x0_glob = numpy.reshape(init[init_idx], (self.numJoints, 3))*self.Di_scale[init_idx]+self.Di_off3D[init_idx]
                # x0 = ((x0_glob - self.Di_off3D[idx1])/self.Di_scale[idx1]).flatten()
                if not numpy.allclose(off3D, 0.):
                    # use offset in 3D wrt to the reference frame
                    x0 = (init[ref_idx] - off3D/self.Di_scale[i]).flatten()
                elif not numpy.allclose(off2D, 0.):
                    # use offset in 2D wrt to the reference frame
                    x02D = self.project3Dto2D(init[ref_idx], ref_idx).reshape(self.numJoints, 2)
                    x02D -= off2D/(self.Di.shape[3]/2.)
                    dptinit = init[ref_idx].reshape(self.numJoints, 3)[:, 2].copy()
                    # denormalize depth
                    if self.normZeroOne is True:
                        dptinit *= (2.*self.Di_scale[ref_idx])
                        dptinit += (self.Di_off3D[ref_idx, 2] - self.Di_scale[ref_idx])
                    else:
                        dptinit *= self.Di_scale[ref_idx]
                        dptinit += self.Di_off3D[ref_idx, 2]
                    x0 = self.project2Dto3D(numpy.concatenate([x02D, dptinit.reshape((self.numJoints, 1))], axis=1), ref_idx).flatten()
                else:
                    x0 = init[ref_idx].flatten()
            else:
                i = isa
                ref_idx = int(init_aligned[i, 0, 0])
                x0 = init_aligned[i, 1:, :]

            # pose of closest frame
            init_closest[i, 0] = ref_idx
            init_closest[i, 1:, :] = self.li3D_aug[numpy.where(self.subset_idxs == ref_idx)[0][0]].reshape(init_closest[0, 1:, :].shape)

            # aligned pose
            init_aligned[i, 0] = ref_idx
            init_aligned[i, 1:, :] = x0.reshape(init_closest[0, 1:, :].shape)

            if self.isli2D:
                li_denorm = (self.li * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
            else:
                li_denorm = (self.project3Dto2D(self.li, self.subset_idxs) * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
            # skip reference frames
            if i in self.subset_idxs:
                init[i] = self.li3D_aug[numpy.where(self.subset_idxs == i)[0][0]].reshape(init[0].shape)
                continue

            li_idx = numpy.where(self.subset_idxs == ref_idx)[0][0]
            for j in range(self.numJoints):
                y_k = numpy.rint(li_denorm[li_idx].reshape((self.numJoints, 2))[j, 1]).astype(int).clip(0, self.Di.shape[2]-1)
                x_k = numpy.rint(li_denorm[li_idx].reshape((self.numJoints, 2))[j, 0]).astype(int).clip(0, self.Di.shape[3]-1)
                ystart_k = y_k - self.corrPatchSize // 2
                yend_k = ystart_k+self.corrPatchSize
                xstart_k = x_k - self.corrPatchSize // 2
                xend_k = xstart_k+self.corrPatchSize

                # create template patches and pad them
                templ_k = self.Di[ref_idx, 0, max(ystart_k, 0):min(yend_k, self.Di.shape[2]),
                                  max(xstart_k, 0):min(xend_k, self.Di.shape[3])].copy()
                templ_k = numpy.pad(templ_k, ((abs(ystart_k)-max(ystart_k, 0), abs(yend_k)-min(yend_k, self.Di.shape[2])),
                                              (abs(xstart_k)-max(xstart_k, 0), abs(xend_k)-min(xend_k, self.Di.shape[3]))),
                                    mode='constant', constant_values=(1.,))

                # mask to fight background
                msk = numpy.bitwise_and(self.Di[i, 0] < 1.-1e-4, self.Di[i, 0] > -1.+1e-4)
                msk = scipy.ndimage.binary_erosion(msk, structure=numpy.ones((3, 3)), iterations=1)
                msk_dt = numpy.bitwise_not(msk)
                edt = scipy.ndimage.morphology.distance_transform_edt(msk_dt)
                edt = cv2.normalize(edt, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                # normalized correlation maps
                di_pad = numpy.pad(self.Di[i, 0].astype('float32'), ((self.corrPatchSize // 2, self.corrPatchSize // 2-1),
                                                                     (self.corrPatchSize // 2, self.corrPatchSize // 2-1)),
                                   mode='constant', constant_values=(1.,))
                corrMaps_k[0, j] = cv2.matchTemplate(di_pad, templ_k.astype('float32'), self.corrMethod)
                corrMaps_k[0, j] = cv2.normalize(corrMaps_k[0, j], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                corrMaps_k[0, j] *= msk.astype(corrMaps_k.dtype)
                # correlation is high where similar, we minimize dissimilarity which is 1-corr
                corrMaps_k[0, j] = 1. - corrMaps_k[0, j] + numpy.square(edt + 1.)*msk_dt.astype(corrMaps_k.dtype)

                # scipy.misc.imshow(templ_k)
                # scipy.misc.imshow(templ_kp1)
                # scipy.misc.imshow(corrMaps_k[i,j])
                # scipy.misc.imshow(corrMaps_kp1[i,j])
                # print cv2.minMaxLoc(corrMaps_k[i,j]), x_k, y_k
                # print cv2.minMaxLoc(corrMaps_kp1[i,j]), x_kp1, y_kp1

            def train_fn_jaco(x):
                # |y_i - f(x_i)|^2
                # J = df(x)/dx
                row = []
                col = []
                data = []
                offset = 0

                # correlation maps
                Li_img2D = (self.project3Dto2D(x, i) * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
                Li_img2D = Li_img2D.reshape((1, self.numJoints, 2))
                # bilinear interpolation on correlation maps
                j = numpy.ogrid[0:self.numJoints]
                x1 = numpy.floor(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
                y1 = numpy.floor(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
                x2 = numpy.ceil(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
                y2 = numpy.ceil(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
                # div by zero error if x1 = x2 = x if x in Z
                x2[numpy.where(x1 == x2)] += 1
                y2[numpy.where(y1 == y2)] += 1
                xx = Li_img2D[:, :, 0]
                yy = Li_img2D[:, :, 1]
                fQ11_k = corrMaps_k[0, j, y1, x1]
                fQ21_k = corrMaps_k[0, j, y1, x2]
                fQ12_k = corrMaps_k[0, j, y2, x1]
                fQ22_k = corrMaps_k[0, j, y2, x2]
                grad_x = 1./(x2-x1)/(y2-y1)*(-fQ11_k*(y2-yy) + fQ21_k*(y2-yy) - fQ12_k*(yy-y1) + fQ22_k*(yy-y1))
                grad_y = 1./(x2-x1)/(y2-y1)*(-fQ11_k*(x2-xx) - fQ21_k*(xx-x1) + fQ12_k*(x2-xx) + fQ22_k*(xx-x1))

                # optimize only one sample
                for k in xrange(self.numJoints):
                    row.extend([k+offset,
                                k+offset,
                                k+offset])
                    col.extend([k*3+0,
                                k*3+1,
                                k*3+2])

                    s = self.Di_scale[i]
                    cx = self.Di_off3D[i][0]
                    cy = self.Di_off3D[i][1]
                    cz = self.Di_off3D[i][2]
                    px = x[k*3+0]
                    py = x[k*3+1]
                    pz = x[k*3+2]
                    kx = self.Di_trans2D[i][0, 0]
                    ky = self.Di_trans2D[i][1, 1]
                    fx = self.cam_proj[0, 0]
                    fy = self.cam_proj[1, 1]
                    norm = (self.Di.shape[3]/2.)
                    data.extend([grad_x[0, k] * kx*fx*s/(s*pz+cz)/norm*self.init_lambdaP/self.numJoints,
                                 grad_y[0, k] * ky*fy*s/(s*pz+cz)/norm*self.init_lambdaP/self.numJoints,
                                 (grad_x[0, k] * -kx*fx*(s*px+cx)*s/(s*pz+cz)**2. + grad_y[0, k] * -ky*fy*(s*py+cy)*s/(s*pz+cz)**2.)/norm*self.init_lambdaP/self.numJoints])

                offset += 1*self.numJoints
                # hard constraint, off-diagonal
                lag = tsMuLagr.get_value()
                for k in xrange(len(self.lenghtConstraintIdx)):
                    row.extend([k+offset,
                                k+offset,
                                k+offset,
                                k+offset,
                                k+offset,
                                k+offset])
                    col.extend([self.lenghtConstraintIdx[k][0]*3+0,
                                self.lenghtConstraintIdx[k][0]*3+1,
                                self.lenghtConstraintIdx[k][0]*3+2,
                                self.lenghtConstraintIdx[k][1]*3+0,
                                self.lenghtConstraintIdx[k][1]*3+1,
                                self.lenghtConstraintIdx[k][1]*3+2])
                    data.extend([2.*(x[self.lenghtConstraintIdx[k][0]*3+0] - x[self.lenghtConstraintIdx[k][1]*3+0])*lag/len(self.lenghtConstraintIdx),
                                 2.*(x[self.lenghtConstraintIdx[k][0]*3+1] - x[self.lenghtConstraintIdx[k][1]*3+1])*lag/len(self.lenghtConstraintIdx),
                                 2.*(x[self.lenghtConstraintIdx[k][0]*3+2] - x[self.lenghtConstraintIdx[k][1]*3+2])*lag/len(self.lenghtConstraintIdx),
                                 2.*(x[self.lenghtConstraintIdx[k][1]*3+0] - x[self.lenghtConstraintIdx[k][0]*3+0])*lag/len(self.lenghtConstraintIdx),
                                 2.*(x[self.lenghtConstraintIdx[k][1]*3+1] - x[self.lenghtConstraintIdx[k][0]*3+1])*lag/len(self.lenghtConstraintIdx),
                                 2.*(x[self.lenghtConstraintIdx[k][1]*3+2] - x[self.lenghtConstraintIdx[k][0]*3+2])*lag/len(self.lenghtConstraintIdx)])

                offset += 1*len(self.lenghtConstraintIdx)
                nvars = 1*self.numJoints*3
                spmat = sps.coo_matrix((data, (row, col)), shape=(offset, nvars), dtype=float)

                # import matplotlib.pyplot as plt
                # plt.clf()
                # plt.spy(spmat, marker='.', markersize=2)
                # plt.show(block=True)

                return spmat

            def train_fn_err(x):
                # y_i - f(x_i)
                Li_img2D = (self.project3Dto2D(x, i) * (self.Di.shape[3]/2.)) + (self.Di.shape[3]/2.)
                Li_img2D = Li_img2D.reshape((1, self.numJoints, 2))
                # bilinear interpolation on correlation maps
                j = numpy.ogrid[0:self.numJoints]
                x1 = numpy.floor(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
                y1 = numpy.floor(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
                x2 = numpy.ceil(Li_img2D[:, :, 0]).astype(int).clip(0, self.Di.shape[3]-2)
                y2 = numpy.ceil(Li_img2D[:, :, 1]).astype(int).clip(0, self.Di.shape[2]-2)
                # div by zero error if x1 = x2 = x if x in Z
                x2[numpy.where(x1 == x2)] += 1
                y2[numpy.where(y1 == y2)] += 1
                xx = Li_img2D[:, :, 0]
                yy = Li_img2D[:, :, 1]
                fQ11_k = corrMaps_k[0, j, y1, x1]
                fQ21_k = corrMaps_k[0, j, y1, x2]
                fQ12_k = corrMaps_k[0, j, y2, x1]
                fQ22_k = corrMaps_k[0, j, y2, x2]
                val_k = 1./(x2-x1)/(y2-y1)*(fQ11_k*(x2-xx)*(y2-yy) + fQ21_k*(xx-x1)*(y2-yy) + fQ12_k*(x2-xx)*(yy-y1) + fQ22_k*(xx-x1)*(yy-y1))

                err = numpy.concatenate([self.init_lambdaP/self.numJoints*(0.-val_k).ravel(),
                                         tsMuLagr.get_value()/len(self.lenghtConstraintIdx)*(numpy.square(self.jointLength[self.hcIdxForFlatIdx(i)]) - numpy.dot(numpy.square(numpy.dot(x.reshape(1, self.numJoints*3), self.lenghtConstraint)), self.lenghtConstraintS)).ravel()])
                return err

            # creates a function that computes the cost
            def train_fn(x):
                return numpy.square(train_fn_err(x)).sum()

            print "CURRENT: ", i, "INIT: ", ref_idx
            assert not numpy.allclose(x0, 0.)
            if 'init_optimize_incHC' in self.eval_params:
                if self.eval_params['init_optimize_incHC'] is True:
                    tsMuLagr.set_value(numpy.cast['float32'](self.init_muLagr))
                    best_rval = [x0]
                    for mi in range(5):
                        best_rval = sparseLM(x0=best_rval[0],
                                             func=train_fn,
                                             fjaco=train_fn_jaco,
                                             ferr=train_fn_err,
                                             max_iter=100,
                                             eps_grad=1e-11,
                                             eps_param=1e-8,
                                             eps_cost=1e-8,
                                             eps_improv=1e-8,
                                             retall=True,
                                             disp=0)
                        tsMuLagr.set_value(numpy.cast['float32'](tsMuLagr.get_value()*10.))
                else:
                    best_rval = sparseLM(x0=x0,
                                         func=train_fn,
                                         fjaco=train_fn_jaco,
                                         ferr=train_fn_err,
                                         max_iter=100,
                                         eps_grad=1e-11,
                                         eps_param=1e-8,
                                         eps_cost=1e-8,
                                         eps_improv=1e-8,
                                         retall=True,
                                         disp=0)
            else:
                raise NotImplementedError("Missing parameter 'init_optimize_incHC'")

            init[i] = best_rval[0].reshape(init[0].shape)

            if 'init_manualrefinement' in self.eval_params:
                if self.eval_params['init_manualrefinement'] is True:
                    init[i], ncor = self.checkInitAccuracy(init[i], i)
                    if len(ncor) > 0:
                        corrected.append([i, ncor])
            else:
                raise NotImplementedError("Missing parameter 'init_manualrefinement'")

            if 'init_incrementalref' in self.eval_params:
                if self.eval_params['init_incrementalref'] is True:
                    if hasattr(self, 'ref_emb') and hasattr(self, 'sample_emb'):
                        if i not in self.subset_idxs:
                            sidx = numpy.where(unannotated == i)[0][0]
                            # add element to reference frames
                            self.li = numpy.vstack((self.li, self.project3Dto2D(init[i], i).reshape(1, self.numJoints*2)))
                            self.subset_idxs = numpy.append(self.subset_idxs, i)
                            self.numSubset += 1
                            self.addPBVisForLi(i)
                            self.li3D_aug = numpy.vstack((self.li3D_aug, init[i].reshape(1, self.numJoints, 3)))
                            self.ref_emb = numpy.vstack((self.ref_emb, self.sample_emb[sidx][None, :]))
                            self.sample_emb = numpy.delete(self.sample_emb, sidx, axis=0)
                            if self.eval_params['ref_descriptor'] == 'hog_msk':
                                raise NotImplementedError("")
                                self.ref_mask = numpy.vstack((self.ref_mask, sample_mask[sidx][None, :]))
                    else:
                        raise NotImplementedError("ref_emb or sample_emb not set!")
            else:
                raise NotImplementedError("Missing parameter 'init_incrementalref'")

            pbar.update(isa)

            # show initialization
            if self.debugPrint:
                li2D = self.project3Dto2D(init[i], i)*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)
                x02D = self.project3Dto2D(init[ref_idx], ref_idx)*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)
                x02D2 = self.project3Dto2D(x0, ref_idx)*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)
                if self.gt3D is not None:
                    gt2D = self.project3Dto2D(self.gt3D[i], i)*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)
                else:
                    gt2D = numpy.zeros_like(x02D)
                if self.normZeroOne is True:
                    dpt = self.Di[i, 0]*(2.*self.Di_scale[i]) + (self.Di_off3D[i, 2] - self.Di_scale[i])
                    ref_dpt = self.Di[ref_idx, 0]*(2.*self.Di_scale[ref_idx]) + (self.Di_off3D[ref_idx, 2] - self.Di_scale[ref_idx])
                else:
                    dpt = self.Di[i, 0]*self.Di_scale[i] + self.Di_off3D[i, 2]
                    ref_dpt = self.Di[ref_idx, 0]*self.Di_scale[ref_idx] + self.Di_off3D[ref_idx, 2]
                dpt[dpt == (self.Di_off3D[i, 2] + self.Di_scale[i])] = 0.
                ref_dpt[ref_dpt == (self.Di_off3D[ref_idx, 2] + self.Di_scale[ref_idx])] = 0.
                comb = numpy.concatenate([self.hpe.plotResult(ref_dpt, x02D.reshape((self.numJoints, 2)),
                                                              x02D.reshape((self.numJoints, 2)), niceColors=True),
                                          self.hpe.plotResult(dpt, x02D.reshape((self.numJoints, 2)),
                                                              x02D.reshape((self.numJoints, 2)), niceColors=True),
                                          self.hpe.plotResult(dpt, x02D2.reshape((self.numJoints, 2)),
                                                              x02D2.reshape((self.numJoints, 2)), niceColors=True),
                                          self.hpe.plotResult(dpt, li2D.reshape((self.numJoints, 2)),
                                                              li2D.reshape((self.numJoints, 2)), niceColors=True),
                                          self.hpe.plotResult(dpt, gt2D.reshape((self.numJoints, 2)),
                                                              gt2D.reshape((self.numJoints, 2)), niceColors=True)],
                                         axis=1)

                cv2.putText(comb, 'init on ref, init on cur, align on cur, optimized on cur, GT on cur', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255))
                cv2.imwrite(self.eval_prefix+'/init_closest_{}_{}.png'.format(self.eval_params['init_offset'], i), comb)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(comb)
                plt.show(block=True)
                plt.close(fig)

                self.plotCorrMaps(corrMaps_k[0])

                # mj = numpy.square(init[i]-self.gt3D[i]).sum(axis=1).argmax(axis=0)
                # print "WORST: ", mj
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # ax.imshow(corrMaps_k[0, mj])
                # plt.show(block=True)
                # plt.close(fig)
                #
                # self.hpe.plotResult3D(dpt, self.Di_trans2D[i].T,
                #                  init[i].reshape((self.numJoints, 3))*self.Di_scale[i]+self.Di_off3D[i],
                #                  init[i].reshape((self.numJoints, 3))*self.Di_scale[i]+self.Di_off3D[i],
                #                  niceColors=True)

        pbar.finish()

        print "Took {}s for init closest reference".format(time.time()-start)

        # save intermediary results
        numpy.save("{}/Li_init_{}_ref.npy".format(self.eval_prefix, self.eval_params['init_method']), init_closest)
        numpy.save("{}/Li_init_{}_aligned_{}.npy".format(self.eval_prefix, self.eval_params['init_method'],
                                                         self.eval_params['init_offset']), init_aligned)

        print "Corrected: #{}, {}".format(len([c for frame in corrected for c in frame[1]]), corrected)
        f = open("{}/Li_init_{}_corrected_{}.txt".format(self.eval_prefix, self.eval_params['init_method'],
                                                         self.eval_params['init_offset']), 'w')
        for i in corrected:
            f.write("{}, {}\n".format(i[0], ', '.join(map(str, i[1].tolist()))))

        return init.reshape((self.numSamples, self.numJoints, 3)), corrected

    def checkInitAccuracy(self, li3D, idx):
        """
        Checks the initialization accuracy
        :param li3D: normalized 3D annotations
        :param idx: index of annotation
        :return: corrected annotation
        """

        jcorr = []

        if self.gt3D is not None:
            # automatic check to simulate user interaction
            if 'init_accuracy_tresh' in self.eval_params:
                thresh = float(self.eval_params['init_accuracy_tresh']) / (self.Di.shape[3]/2.)
            else:
                raise NotImplementedError("Missing parameter 'init_accuracy_tresh'")
            gt2D = self.project3Dto2D(self.gt3D[idx], idx).reshape(self.numJoints, 2)
            li2D = self.project3Dto2D(li3D, idx).reshape(self.numJoints, 2)

            diffs = numpy.sqrt(numpy.square(li2D - gt2D).sum(axis=1))
            if diffs.max() > thresh:
                jcorr = numpy.where(diffs > thresh)[0]
                # if self.normZeroOne is True:
                #     dpt = self.Di[idx, 0]*(2.*self.Di_scale[idx]) + (self.Di_off3D[idx, 2] - self.Di_scale[idx])
                # else:
                #     dpt = self.Di[idx, 0]*self.Di_scale[idx] + self.Di_off3D[idx, 2]
                # dpt[dpt == (self.Di_off3D[idx, 2] + self.Di_scale[idx])] = 0.
                # self.hpe.plotResult(dpt, (self.project3Dto2D(self.gt3D[idx], idx)*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)).reshape((self.numJoints, 2)),
                #                (self.project3Dto2D(li3D, idx)*(self.Di.shape[3]/2.) + (self.Di.shape[3]/2.)).reshape((self.numJoints, 2)), niceColors=True)

                # correct 2D locations and use depth as initialization
                li2D = gt2D
                dptinit = li3D[:, 2].reshape(1, self.numJoints).copy()
                # denormalize depth
                if self.normZeroOne is True:
                    dptinit *= (2.*self.Di_scale[idx])
                    dptinit += (self.Di_off3D[idx, 2] - self.Di_scale[idx])
                else:
                    dptinit *= self.Di_scale[idx]
                    dptinit += self.Di_off3D[idx, 2]
                # add element to reference frames
                unannotated = numpy.setdiff1d(numpy.arange(self.numSamples), self.subset_idxs, assume_unique=True)
                self.li = numpy.vstack((self.li, li2D.reshape(1, self.numJoints*2)))
                self.subset_idxs = numpy.append(self.subset_idxs, idx)
                self.numSubset += 1
                self.addPBVisForLi(idx)
                new_li3D = self.augmentLi3D(li2D.reshape(1, self.numJoints, 2), idx, dptinit)
                self.li3D_aug = numpy.vstack((self.li3D_aug, new_li3D))

                if hasattr(self, 'ref_emb') and hasattr(self, 'sample_emb'):
                    sidx = numpy.where(unannotated == idx)[0][0]
                    self.ref_emb = numpy.vstack((self.ref_emb, self.sample_emb[sidx][None, :]))
                    self.sample_emb = numpy.delete(self.sample_emb, sidx, axis=0)
                    if self.eval_params['ref_descriptor'] == 'hog_msk':
                        raise NotImplementedError("")
                        self.ref_mask = numpy.vstack((self.ref_mask, new_mask))
                else:
                    raise NotImplementedError("ref_emb and sample_emb not set!")

                print "Corrected: ", jcorr
                print "New accuracy: ", self.evaluateToGT(new_li3D.reshape(1, self.numJoints, 3), [idx])

                return new_li3D.reshape(li3D.shape), jcorr
            else:
                print "OK"
                # nothing to change
                return li3D, jcorr
        else:
            # user interaction
            import matplotlib.pyplot as plt
            if plt.matplotlib.get_backend() == 'agg':
                raise EnvironmentError("Need matplotlib in interactive mode, not background mode!")

            li2D = self.project3Dto2D(li3D, idx).reshape(self.numJoints, 2)*(self.Di.shape[3]/2.)+(self.Di.shape[3]/2.)
            if self.normZeroOne is True:
                dpt = self.Di[idx, 0]*(2.*self.Di_scale[idx]) + (self.Di_off3D[idx, 2] - self.Di_scale[idx])
            else:
                dpt = self.Di[idx, 0]*self.Di_scale[idx] + self.Di_off3D[idx, 2]
            dpt[dpt == (self.Di_off3D[idx, 2] + self.Di_scale[idx])] = 0.
            self.hpe.plotResult(dpt, li2D, li2D, showGT=False, niceColors=True)

            ok = raw_input('Annotation OK? Press y or n!\n').strip()
            while ok != 'y' and ok != 'n':
                ok = raw_input('Annotation OK? Press y or n!\n').strip()

            if ok == 'n':
                from PyQt4.QtGui import QApplication
                from util.interactivedatasetlabeling import InteractiveDatasetLabeling
                from data.basetypes import NamedImgSequence, DepthFrame
                import copy
                # create dummy sequence
                if self.normZeroOne is True:
                    dpt = self.Di[idx, 0]*(2.*self.Di_scale[idx]) + (self.Di_off3D[idx, 2] - self.Di_scale[idx])
                else:
                    dpt = self.Di[idx, 0]*self.Di_scale[idx] + self.Di_off3D[idx, 2]
                dpt[dpt == (self.Di_off3D[idx, 2] + self.Di_scale[idx])] = 0.
                li_img2D_hom = numpy.concatenate([li2D[:, 0:2], numpy.ones((self.numJoints, 1))], axis=1)
                li_glob2D = numpy.dot(li_img2D_hom, numpy.linalg.inv(self.Di_trans2D[idx]))
                li_glob2D = li_glob2D[:, 0:2] / li_glob2D[:, 2][:, None]
                frame = DepthFrame(dpt.astype(numpy.float32), numpy.concatenate([li_glob2D, li3D[:, 2][:, None]*self.Di_scale[idx] + self.Di_off3D[idx][2]], axis=1),
                                   numpy.concatenate([li2D, li3D[:, 2][:, None]*self.Di_scale[idx] + self.Di_off3D[idx][2]], axis=1),
                                   self.Di_trans2D[idx].transpose(), li3D*self.Di_scale[idx]+self.Di_off3D[idx], li3D*self.Di_scale[idx],
                                   self.Di_off3D[idx], self.depthFiles[idx], '', '', {'vis': [], 'pb': {'pb': [], 'pbp': []}})
                seq = NamedImgSequence('temp', [frame], {'cube': (self.Di_scale[idx]*2., self.Di_scale[idx]*2., self.Di_scale[idx]*2.)})
                # We are only allowed to create a single instance of QApplication, so use lazy init
                if not hasattr(self, 'qt_interactive_app'):
                    self.qt_interactive_app = QApplication(sys.argv)
                browser = InteractiveDatasetLabeling(seq, self.hpe, self.importer, self.hc, None, None, None, None, [], 0)
                browser.show()
                self.qt_interactive_app.exec_()
                print browser.curData, browser.curVis, browser.curPb, browser.curPbP, browser.curCorrected
                jcorr = copy.deepcopy(browser.curCorrected)
                gt2D = browser.curData.copy()
                vis = copy.deepcopy(browser.curVis)
                pb = copy.deepcopy(browser.curPb)
                pbp = copy.deepcopy(browser.curPbP)
                del browser
                gc.collect()
                gc.collect()
                gc.collect()

                # correct 2D locations
                Li_glob2D = numpy.concatenate([gt2D[:, 0:2], numpy.ones((self.numJoints, 1))], axis=1)
                Li_img2D_hom = numpy.dot(Li_glob2D, self.Di_trans2D[idx])
                Li_img2D = (Li_img2D_hom[:, 0:2] / Li_img2D_hom[:, 2][:, None]).reshape((self.numJoints*2))
                li2D = (Li_img2D - (self.Di.shape[3]/2.)) / (self.Di.shape[3]/2.)
                # add element to reference frames
                unannotated = numpy.setdiff1d(numpy.arange(self.numSamples), self.subset_idxs, assume_unique=True)
                self.li = numpy.vstack((self.li, li2D.reshape(1, self.numJoints*2)))
                self.subset_idxs = numpy.append(self.subset_idxs, idx)
                self.numSubset += 1
                self.addPBVisForLi(idx, vis=vis, pb=pb, pbp=pbp)
                new_li3D = self.augmentLi3D(li2D.reshape(1, self.numJoints, 2), idx)
                self.li3D_aug = numpy.vstack((self.li3D_aug, new_li3D))

                if hasattr(self, 'ref_emb') and hasattr(self, 'sample_emb'):
                    sidx = numpy.where(unannotated == idx)[0][0]
                    self.ref_emb = numpy.vstack((self.ref_emb, self.sample_emb[sidx][None, :]))
                    self.sample_emb = numpy.delete(self.sample_emb, sidx, axis=0)
                    if self.eval_params['ref_descriptor'] == 'hog_msk':
                        raise NotImplementedError("")
                        self.ref_mask = numpy.vstack((self.ref_mask, new_mask))
                else:
                    raise NotImplementedError("ref_emb and sample_emb not set!")

                # new_li2D = self.project3Dto2D(new_li3D, idx).reshape(self.numJoints, 2)*(self.Di.shape[3]/2.)+(self.Di.shape[3]/2.)
                # self.hpe.plotResult(dpt, new_li2D, new_li2D, showGT=False, niceColors=True)

                return new_li3D.reshape(li3D.shape), jcorr
            else:
                print "OK"
                # nothing to change
                return li3D, jcorr

    def addPBVisForLi(self, idx, orig=False, vis=None, pb=None, pbp=None):

        if self.gt3D is None:
            if vis is None:
                print "WARNING: Cannot automatically determine without ground truth!"
                li_visiblemask = numpy.zeros((self.numJoints,))
            else:
                li_visiblemask = numpy.ones((self.numJoints,))
                # remove not visible ones
                occluded = numpy.setdiff1d(numpy.arange(self.numJoints), vis)
                li_visiblemask[occluded] = 0
        else:
            # check if visibility is used
            if numpy.allclose(self.li_visiblemask, 0):
                li_visiblemask = numpy.zeros((self.numJoints,))
            else:
                li_visiblemask = numpy.ones((self.numJoints,))
                dpt = self.importer.loadDepthMap(self.depthFiles[idx])
                Li_glob3D = (numpy.reshape(self.gt3D[idx], (self.numJoints, 3))*self.Di_scale[idx]+self.Di_off3D[idx][None, :]).reshape((self.numJoints, 3))
                Li_glob3D_hom = numpy.concatenate([Li_glob3D, numpy.ones((self.numJoints, 1), dtype='float32')], axis=1)
                Li_glob2D_hom = numpy.dot(Li_glob3D_hom, self.cam_proj.T)
                Li_glob2D = (Li_glob2D_hom[:, 0:3] / Li_glob2D_hom[:, 3][:, None]).reshape((self.numJoints, 3))
                Li_glob2D[:, 2] = Li_glob3D[:, 2]
                vis = self.importer.visibilityTest(dpt, Li_glob2D, 10.)
                # remove not visible ones
                occluded = numpy.setdiff1d(numpy.arange(self.numJoints), vis)
                li_visiblemask[occluded] = 0
        self.li_visiblemask = numpy.vstack((self.li_visiblemask, li_visiblemask))
        if orig:
            self.orig_li_visiblemask = numpy.vstack((self.orig_li_visiblemask, li_visiblemask))

        if self.gt3D is None:
            if pb is None or pbp is None:
                print "WARNING: Cannot automatically determine without ground truth!"
                lip = []
                pip = []
            else:
                lip = pb
                pip = pbp
        else:
            lip = []
            pip = []
            for p in range(len(self.posebits)):
                if abs(self.gt3D[idx, self.posebits[p][0], 2] - self.gt3D[idx, self.posebits[p][1], 2]) > self.pb_thresh/self.Di_scale[idx]:
                    if self.gt3D[idx, self.posebits[p][0], 2] < self.gt3D[idx, self.posebits[p][1], 2]:
                        lip.append((self.posebits[p][0], self.posebits[p][1]))
                        pip.append(self.posebits[p])
                    else:
                        lip.append((self.posebits[p][1], self.posebits[p][0]))
                        pip.append(self.posebits[p])
        self.li_posebits.append(lip)
        self.li_posebitIdx.append(pip)
        if orig:
            self.orig_li_posebits.append(lip)
            self.orig_li_posebitIdx.append(pip)

    def plotErrors(self, allerrsLi, allerrsLiref, allerrsWK, allerrsWKref, allcosts, leg):
        """
        Creates error plots over epochs
        :param allerrsLi: errors on all 3D poses
        :param allerrsLiref: errors on the reference 3D poses
        :param allerrsWK: errors of the regression for all poses
        :param allerrsWKref: errors of the regression for the reference frames
        :param allcosts: all costs of the objective function
        :return: None
        """

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([ac[0] for ac in allerrsLi])
        ax.plot([ac[0] for ac in allerrsWK])
        ax.set_yscale('symlog')
        ax.grid('on')
        lgd = ax.legend(['Avg. error all Li', 'Avg. error WK all Li'], loc='upper center',
                        bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.show(block=False)
        fig.savefig(self.eval_prefix+'/meanerrs.pdf', bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([ac[1] for ac in allerrsLi])
        ax.plot([ac[1] for ac in allerrsWK])
        ax.set_yscale('symlog')
        ax.grid('on')
        lgd = ax.legend(['Max error all Li', 'Max error WK all Li'], loc='upper center',
                        bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.show(block=False)
        fig.savefig(self.eval_prefix+'/maxerrs.pdf', bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([ac[0] for ac in allerrsLiref])
        ax.plot([ac[0] for ac in allerrsWKref])
        ax.set_yscale('symlog')
        ax.grid('on')
        lgd = ax.legend(['Avg. error ref Li', 'Avg. error WK ref Li'], loc='upper center',
                        bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.show(block=False)
        fig.savefig(self.eval_prefix+'/meanerrs_ref.pdf', bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([ac[1] for ac in allerrsLiref])
        ax.plot([ac[1] for ac in allerrsWKref])
        ax.set_yscale('symlog')
        ax.grid('on')
        lgd = ax.legend(['Max error ref Li', 'Max error WK ref Li'], loc='upper center',
                        bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.show(block=False)
        fig.savefig(self.eval_prefix+'/maxerrs_ref.pdf', bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([ac[0] for ac in allcosts])
        ax.set_yscale('symlog')
        ax.grid('on')
        plt.show(block=False)
        fig.savefig(self.eval_prefix+'/allcosts.pdf')
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(numpy.asarray(allcosts))
        ax.set_yscale('symlog')
        ax.grid('on')
        lgd = ax.legend(leg, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.show(block=False)
        fig.savefig(self.eval_prefix+'/costs.pdf', bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)

    def plotCorrMaps(self, corrMap):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        r = c = int(numpy.ceil(numpy.sqrt(float(corrMap.shape[0]))))
        for i in range(corrMap.shape[0]):
            ax = fig.add_subplot(r, c, i+1)
            ax.imshow(numpy.minimum(corrMap[i], 1.), interpolation='none')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show(block=True)
        plt.close(fig)

    def hcIdxForFlatIdx(self, i):
        # hc breaks are inclusive
        assert i <= max(self.hcBreaks)
        nums = numpy.insert(numpy.cumsum(self.hcBreaks), 0, 0)
        d1 = nums - i
        d1[d1 >= 0] = -max(self.hcBreaks)
        hcidx = numpy.argmax(d1)
        return hcidx

