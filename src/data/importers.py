"""Provides importer classes for importing data from different datasets.

DepthImporter provides interface for loading the data from a dataset, esp depth images.
ICVLImporter, NYUImporter, MSRAImporter are specific instances of different importers.

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
import copy

import fnmatch
import scipy.io
import scipy.signal
import numpy as np
from PIL import Image
import glob
import os

import cv2
import progressbar as pb
import struct
from data.basetypes import DepthFrame, NamedImgSequence
from util.handdetector import HandDetector
from data.transformations import transformPoint2D
import cPickle

__author__ = "Paul Wohlhart <wohlhart@icg.tugraz.at>, Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class DepthImporter(object):
    """
    provide baisc functionality to load depth data
    """

    def __init__(self, fx, fy, ux, uy):
        """
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """

        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x3 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        raise NotImplementedError("Must be overloaded by base!")

    @staticmethod
    def depthToPCL(dpt, T, background_val=0.):

        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]], np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[np.where(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - 160.) / 241.42 * depth
        col = (pts[:, 1] - 120.) / 241.42 * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

    @staticmethod
    def visibilityTest(dpt, loc, tol=2.):
        """
        z-buffer like visibility test for non-occluded joints
        :param dpt: depth image
        :param loc: 2D joint locations
        :param tol: tolerance
        :return: list of indices of visible ie non-occluded joints
        """
        vis = []
        for i in range(loc.shape[0]):
            y = np.rint(loc[i, 1]).astype(int)
            x = np.rint(loc[i, 0]).astype(int)
            if 0 <= x < dpt.shape[1] and 0 <= y < dpt.shape[0]:
                if np.fabs(dpt[y, x] - loc[i, 2]) < tol:
                    vis.append(i)
                # else:
                #     print("joint {}: dpt {} anno {}".format(i, dpt[y, x], gtcrop[i, 2]))

        return vis

    def loadSequences(self,seqNames):
        """
        Load multiple sequences
        :param seqNames: sequence names as list
        :returns: list of NamedImgSequence
        """
        data = []

        for seqName in seqNames:
            seqData = self.loadSequence(seqName)
            data.append(seqData)

        return data

    def replaceAnnotations(self, seq, new_gtorig2D):
        """
        Replace annotations of sequence with new data from file
        :param seq: sequence
        :param new_gtorig2D: array with new 2D annotations
        :return: new sequence
        """

        if len(seq.data) != new_gtorig2D.shape[0]:
            raise ValueError("Shape misfit: sequence has {} entries, but new annotation only {}".format(len(seq.data), new_gtorig2D.shape[0]))

        if seq.data[0].gtorig.size != new_gtorig2D[0].size:
            print "WARNING: Size misfit: seq has {}, but new annotations have {}".format(seq.data[0].gtorig.size,
                                                                                         new_gtorig2D[0].size)

        new_gtorig2D = new_gtorig2D.reshape((len(seq.data), seq.data[0].gtorig.shape[0], seq.data[0].gtorig.shape[1]))
        for idx, el in enumerate(seq.data):
            new_gtcrop = np.zeros((el.gtorig.shape[0], 3), np.float32)
            for joint in xrange(el.gtorig.shape[0]):
                t = transformPoint2D(new_gtorig2D[idx][joint], el.T)
                new_gtcrop[joint, 0] = t[0]
                new_gtcrop[joint, 1] = t[1]
                new_gtcrop[joint, 2] = new_gtorig2D[idx][joint, 2]

            new_gt3Dorig = self.jointsImgTo3D(new_gtorig2D[idx])
            new_gt3Dcrop = new_gt3Dorig - el.com

            seq.data[idx] = el._replace(gtorig=new_gtorig2D[idx],
                                        gtcrop=new_gtcrop,
                                        gt3Dorig=new_gt3Dorig,
                                        gt3Dcrop=new_gt3Dcrop)

        return seq

    def saveSequenceAnnotations(self, seqData, files):
        """
        Saves the annotations of a sequence to file, for joint annotations save gtorig
        :param seqData: NamedImgSequence sequence data
        :param files: dictionary with at least 'joints' file selected
        :return: None
        """

        if not isinstance(seqData, NamedImgSequence):
            raise ValueError("Need sequence as NamedImageSequence")

        if seqData.data[0].extraData:
            for k in seqData.data[0].extraData.keys():
                if k not in files.keys():
                    raise IndexError("Require file name for extra data {}, but got {}".format(k, files.keys()))

        if 'joints' not in files.keys():
            raise IndexError("Require joints file, but got {}".format(files.keys))

        print "Saving {}".format(seqData.name)

        # open needed files
        fhandles = {}
        for f in files.keys():
            fhandles[f] = open(files[f], 'w')
            fhandles[f].write("{}\n".format(len(seqData.data)))

        # save all data
        for seq in seqData.data:
            # save joints
            fhandles['joints'].write(seq.fileName + " ")
            for i in range(seq.gtorig.shape[0]):
                fhandles['joints'].write("{0:.3f} ".format(float(seq.gtorig[i, 0])))
                fhandles['joints'].write("{0:.3f} ".format(float(seq.gtorig[i, 1])))
                fhandles['joints'].write("{0:.3f} ".format(float(seq.gtorig[i, 2])))
            fhandles['joints'].write("\n")

            # save extra data
            if seq.extraData:
                for k in seq.extraData.keys():
                    if k == 'pb':
                        fhandles['pb'].write(seq.fileName + " ")
                        assert len(seq.extraData['pb']['pb']) == len(seq.extraData['pb']['pbp'])
                        for i in range(len(seq.extraData['pb']['pb'])):
                            fhandles['pb'].write("{0:d} {1:d} ".format(int(seq.extraData['pb']['pbp'][i][0]),
                                                                       int(seq.extraData['pb']['pbp'][i][1])))
                            fhandles['pb'].write("{0:d} {1:d} ".format(int(seq.extraData['pb']['pb'][i][0]),
                                                                       int(seq.extraData['pb']['pb'][i][1])))
                        fhandles['pb'].write("\n")
                    elif k == 'vis':
                        fhandles['vis'].write(seq.fileName + " ")
                        for i in range(len(seq.extraData['vis'])):
                            fhandles['vis'].write("{0:d} ".format(int(seq.extraData['vis'][i])))
                        fhandles['vis'].write("\n")
                    elif k == 'subset_anno':
                        if seq.extraData['subset_anno'] is True:
                            fhandles['subset_anno'].write(seq.fileName + "\n")
                    else:
                        raise NotImplementedError("Unknown extraData key: {}".format(k))
            else:
                # write empty entries for completeness
                fhandles['pb'].write(seq.fileName + " ")
                fhandles['pb'].write("\n")
                fhandles['vis'].write(seq.fileName + " ")
                fhandles['vis'].write("\n")

        # close opened files
        for f in fhandles.keys():
            fhandles[f].flush()
            fhandles[f].close()

    def saveSequenceDetections(self, seqData, pfile):
        """
        Saves the detections of a sequence to file, for joint annotations save com in 2D with z
        :param seqData: NamedImgSequence sequence data
        :param pfile: file
        :return: None
        """

        if not isinstance(seqData, NamedImgSequence):
            raise ValueError("Need sequence as NamedImageSequence")

        print "Saving {}".format(seqData.name)

        # open needed file
        fhandle = open(pfile, 'w')
        fhandle.write("{}\n".format(len(seqData.data)))

        # save all data
        for seq in seqData.data:
            # save joints
            fhandle.write(seq.fileName + " ")
            com2D = self.joint3DToImg(seq.com)
            fhandle.write("{0:.3f} {1:.3f} {2:.3f} \n".format(float(com2D[0]), float(com2D[1]), float(com2D[2])))

        fhandle.flush()
        # close opened file
        fhandle.close()

    def poseFromCache(self, cache_name, name, cache_str=''):
        """
        Load hand pose from cached file
        :param cache_name: cache filename
        :param name: hand filename
        :return: pose of hand
        """

        pose = np.zeros_like(self.default_gtorig)
        if cache_str != '':
            count = 0
            for line in cache_str:
                part = line.strip().split(' ')
                if isinstance(name, basestring) and os.path.normpath(part[0]).split(os.path.sep)[-1] == os.path.normpath(name).split(os.path.sep)[-1] and os.path.normpath(part[0]).split(os.path.sep)[-2] == os.path.normpath(name).split(os.path.sep)[-2]:
                     pose = np.asarray(part[1:]).astype(float).reshape((-1, 3))
                if isinstance(name, int) and count == name:
                    pose = np.asarray(part[0:]).astype(float).reshape((-1, 3))
                count += 1
        elif os.path.isfile(cache_name):
            inputfile = open(cache_name, 'r')
            count = 0
            for line in inputfile:
                part = line.strip().split(' ')
                if isinstance(name, basestring) and os.path.normpath(part[0]).split(os.path.sep)[-1] == os.path.normpath(name).split(os.path.sep)[-1] and os.path.normpath(part[0]).split(os.path.sep)[-2] == os.path.normpath(name).split(os.path.sep)[-2]:
                     pose = np.asarray(part[1:]).astype(float).reshape((-1, 3))
                if isinstance(name, int) and count == name:
                    pose = np.asarray(part[0:]).astype(float).reshape((-1, 3))
                count += 1

        if np.allclose(pose, np.zeros_like(self.default_gtorig)):
            print "WARNING: File {} not found in {} or pose not set!".format(name, cache_name)

        return pose

    def visibilityFromCache(self, cache_name, name, cache_str=''):
        """
        Load joint visibility from cached file
        :param cache_name: cache filename
        :param name: hand filename
        :return: visibility for hand
        """

        vis = []

        if cache_str != '':
            count = 0
            for line in cache_str:
                part = line.strip().split(' ')
                if isinstance(name, basestring) and os.path.normpath(part[0]).split(os.path.sep)[-1] == os.path.normpath(name).split(os.path.sep)[-1] and os.path.normpath(part[0]).split(os.path.sep)[-2] == os.path.normpath(name).split(os.path.sep)[-2]:
                    vis = [int(p) for p in part[1:]]
                if isinstance(name, int) and count == name:
                    vis = [int(p) for p in part[0:]]
                count += 1
        elif os.path.isfile(cache_name):
            inputfile = open(cache_name, 'r')
            count = 0
            for line in inputfile:
                part = line.strip().split(' ')
                if isinstance(name, basestring) and os.path.normpath(part[0]).split(os.path.sep)[-1] == os.path.normpath(name).split(os.path.sep)[-1] and os.path.normpath(part[0]).split(os.path.sep)[-2] == os.path.normpath(name).split(os.path.sep)[-2]:
                    vis = [int(p) for p in part[1:]]
                if isinstance(name, int) and count == name:
                    vis = [int(p) for p in part[0:]]
                count += 1

        return vis

    def poseBitsFromCache(self, cache_name, name, cache_str=''):
        """
        Load posebits from cached file
        :param cache_name: cache filename
        :param name: hand filename
        :return: posebits for hand
        """

        pbp = []
        pbi = []

        if cache_str != '':
            count = 0
            for line in cache_str:
                part = line.strip().split(' ')
                if isinstance(name, basestring) and os.path.normpath(part[0]).split(os.path.sep)[-1] == os.path.normpath(name).split(os.path.sep)[-1] and os.path.normpath(part[0]).split(os.path.sep)[-2] == os.path.normpath(name).split(os.path.sep)[-2]:
                    part = part[1:]
                    assert len(part) % 4 == 0
                    # posebits are saved as id1 id2 id1 id2 or id1 id2 id2 id1
                    for k in xrange(len(part)//4):
                        pbp.append((int(part[k*4+0]), int(part[k*4+1])))
                        pbi.append((int(part[k*4+2]), int(part[k*4+3])))
                if isinstance(name, int) and count == name:
                    assert len(part) % 4 == 0
                    # posebits are saved as id1 id2 id1 id2 or id1 id2 id2 id1
                    for k in xrange(len(part)//4):
                        pbp.append((int(part[k*4+0]), int(part[k*4+1])))
                        pbi.append((int(part[k*4+2]), int(part[k*4+3])))
                count += 1
        elif os.path.isfile(cache_name):
            inputfile = open(cache_name, 'r')
            count = 0
            for line in inputfile:
                part = line.strip().split(' ')
                if isinstance(name, basestring) and os.path.normpath(part[0]).split(os.path.sep)[-1] == os.path.normpath(name).split(os.path.sep)[-1] and os.path.normpath(part[0]).split(os.path.sep)[-2] == os.path.normpath(name).split(os.path.sep)[-2]:
                    part = part[1:]
                    assert len(part) % 4 == 0
                    # posebits are saved as id1 id2 id1 id2 or id1 id2 id2 id1
                    for k in xrange(len(part)//4):
                        pbp.append((int(part[k*4+0]), int(part[k*4+1])))
                        pbi.append((int(part[k*4+2]), int(part[k*4+3])))
                if isinstance(name, int) and count == name:
                    assert len(part) % 4 == 0
                    # posebits are saved as id1 id2 id1 id2 or id1 id2 id2 id1
                    for k in xrange(len(part)//4):
                        pbp.append((int(part[k*4+0]), int(part[k*4+1])))
                        pbi.append((int(part[k*4+2]), int(part[k*4+3])))
                count += 1

        return pbi, pbp


class Blender2Importer(DepthImporter):
    """
    provide functionality to load data from the synthetic Blender dataset
    """

    def __init__(self, basepath, useCache=True, cacheDir='./cache/'):
        """
        Constructor
        :param basepath: base path of the MSRA dataset
        :return:
        """

        super(Blender2Importer, self).__init__(460.017965, 459.990440, 320., 240.)

        self.default_gtorig = np.asarray([[22, 65, 0], [48, 55, 0], [74, 28, 0], [87, 13, 0],
                                          [8, 74, 0], [26, -2, 0], [33, -28, 0], [39, -50, 0],
                                          [45, -68, 0], [-4, 71, 0], [0, 0, 0], [-1, -34, 0],
                                          [0, -58, 0], [1, -78, 0], [-17, 71, 0], [-22, 7, 0],
                                          [-28, -21, 0], [-30, -43, 0], [-33, -63, 0], [-25, 70, 0],
                                          [-41, 18, 0], [-46, 0, 0], [-48, -12, 0], [-51, -25, 0]], np.float32)

        self.basepath = basepath
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.numJoints = 24
        self.scales = {'': 1., 'hpseq_loop_mv': 1., 'hpseq_mv': 1., 'hpseq_mv2': 1.,
                       'hpseq': 1., 'hpseq_loop': 1., 'hpseq_nu': 1., 'hpseq_loop_nu': 1.}
        self.sides = {'': 'right', 'hpseq_loop_mv': 'right', 'hpseq_mv': 'right', 'hpseq_mv2': 'right',
                      'hpseq': 'right', 'hpseq_loop': 'right', 'hpseq_nu': 'right', 'hpseq_loop_nu': 'right'}

    def loadDepthMap(self, filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        depth = np.load(filename)
        if len(depth.shape) == 3:
            depth = depth[:, :, 0] * 1000.
        elif len(depth.shape) == 2:
            depth *= 1000.
        else:
            raise IOError("Invalid file: {}".format(filename))
        depth[depth > 1000. - 1e-4] = 32001

        # return in mm
        return depth

    def getDepthMapNV(self):
        """
        Get the value of invalid depth values in the depth map
        :return: value
        """
        return 32001

    def loadSequence(self, seqName, camera=None, Nmax=float('inf'), shuffle=False, rng=None):
        """
        Load an image sequence from the dataset
        :param seqName: sequence name, e.g. subject1
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """

        config = {'cube': (220, 220, 220)}

        pickleCache = '{}/{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, camera)
        if self.useCache & os.path.isfile(pickleCache):
            print("Loading cache data from {}".format(pickleCache))
            f = open(pickleCache, 'rb')
            (seqName, data, config) = cPickle.load(f)
            f.close()
            # shuffle data
            if shuffle and rng is not None:
                print("Shuffling")
                rng.shuffle(data)
            if not (np.isinf(Nmax)):
                return NamedImgSequence(seqName, data[0:Nmax], config)
            else:
                return NamedImgSequence(seqName, data, config)

        # Load the dataset
        objdir = os.path.join(self.basepath, seqName)
        if camera is not None:
            dflt = "c{0:02d}_*.npy".format(camera)
        else:
            dflt = '*.npy'
        dptFiles = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(objdir) for f in
                           fnmatch.filter(files, dflt)])
        # dptFiles = sorted(glob.glob(os.path.join(objdir, '*exr')))

        if camera is not None:
            lflt = "c{0:02d}_*anno_blender.txt".format(camera)
        else:
            lflt = '*anno_blender.txt'
        labelFiles = sorted([os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(objdir) for f in
                             fnmatch.filter(files, lflt)])
        #labelFiles = sorted(glob.glob(os.path.join(objdir, '*txt')))

        # sync lists
        newdpt = []
        newlbl = []
        if camera is not None:
            number_idx = 1
        else:
            number_idx = 0
        labelIdx = [''.join(os.path.basename(lbl).split('_')[number_idx]) for lbl in labelFiles]
        for dpt in dptFiles:
            if ''.join(os.path.basename(dpt).split('_')[number_idx]) in labelIdx:
                newdpt.append(dpt)
                newlbl.append(labelFiles[labelIdx.index(''.join(os.path.basename(dpt).split('_')[number_idx]))])

        dptFiles = newdpt
        labelFiles = newlbl

        assert len(dptFiles) == len(labelFiles)

        txt = 'Loading {}'.format(seqName)
        nImgs = len(dptFiles)
        pbar = pb.ProgressBar(maxval=nImgs, widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()

        data = []
        for i in xrange(nImgs):
            labelFileName = labelFiles[i]
            dptFileName = dptFiles[i]

            if not os.path.isfile(dptFileName):
                print("File {} does not exist!".format(dptFileName))
                continue
            dpt = self.loadDepthMap(dptFileName)

            if not os.path.isfile(labelFileName):
                print("File {} does not exist!".format(labelFileName))
                continue

            # joints in image coordinates
            gtorig = np.genfromtxt(labelFileName)
            gtorig = gtorig.reshape((gtorig.shape[0] / 3, 3))
            gtorig[:, 2] *= 1000.

            # normalized joints in 3D coordinates
            gt3Dorig = self.jointsImgTo3D(gtorig)

            ### Shorting the tip bone, BUGFIX
            tips = [3, 8, 13, 18, 23]
            shrink_factor = 0.85
            for joint_idx in tips:
                t = gt3Dorig[joint_idx, :] - gt3Dorig[joint_idx - 1, :]
                gt3Dorig[joint_idx, :] = gt3Dorig[joint_idx - 1, :] + shrink_factor * t
            gtorig = self.joints3DToImg(gt3Dorig)
            ###

            # print gtorig, gt3Dorig
            # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtorig,0,gt3Dorig,0,0,dptFileName,'','',{}))
            # Detect hand
            hd = HandDetector(dpt, self.fx, self.fy, importer=self)
            if not hd.checkImage(1.):
                print("Skipping image {}, no content".format(dptFileName))
                continue

            try:
                dpt, M, com = hd.cropArea3D(gtorig[10], size=config['cube'])
            except UserWarning:
                print("Skipping image {}, no hand detected".format(dptFileName))
                continue

            com3D = self.jointImgTo3D(com)
            gt3Dcrop = gt3Dorig - com3D  # normalize to com

            gtcrop = np.zeros((gt3Dorig.shape[0], 3), np.float32)
            for joint in range(gtcrop.shape[0]):
                t = transformPoint2D(gtorig[joint], M)
                gtcrop[joint, 0] = t[0]
                gtcrop[joint, 1] = t[1]
                gtcrop[joint, 2] = gtorig[joint, 2]

            # print("{}".format(gt3Dorig))
            # self.showAnnotatedDepth(DepthFrame(dpt,gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,'',''))

            data.append(DepthFrame(dpt.astype(np.float32), gtorig, gtcrop, M, gt3Dorig, gt3Dcrop, com3D, dptFileName,
                                   '', self.sides[seqName], {}))
            pbar.update(i)

            # early stop
            if len(data) >= Nmax:
                break

        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((seqName, data, config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # shuffle data
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return NamedImgSequence(seqName, data, config)

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        import matplotlib
        import matplotlib.pyplot as plt

        print("img min {}, max {}".format(frame.dpt.min(), frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

        ax.plot(frame.gtcrop[0:4, 0], frame.gtcrop[0:4, 1], c='r')
        ax.plot(frame.gtcrop[4:9, 0], frame.gtcrop[4:9, 1], c='r')
        ax.plot(frame.gtcrop[9:14, 0], frame.gtcrop[9:14, 1], c='r')
        ax.plot(frame.gtcrop[14:19, 0], frame.gtcrop[14:19, 1], c='r')
        ax.plot(frame.gtcrop[19:, 0], frame.gtcrop[19:, 1], c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = frame.dpt[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)

        ax.format_coord = format_coord

        for i in range(frame.gtcrop.shape[0]):
            ax.annotate(str(i), (int(frame.gtcrop[i, 0]), int(frame.gtcrop[i, 1])))

        plt.show()

    @staticmethod
    def depthToPCL(dpt, T, background_val=0.):

        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]], np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[np.where(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - 320.) / 460. * depth
        col = (pts[:, 1] - 240.) / 460. * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))


