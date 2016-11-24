"""Provides a basic hand detector in depth images.

HandDetector provides interface for detecting hands in depth image, by using the center of mass.

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
import os
import cv2
from scipy import stats, ndimage

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class HandDetector(object):
    """
    Detect hand based on simple heuristic, centered at Center of Mass
    """

    RESIZE_BILINEAR = 0
    RESIZE_CV2_NN = 1
    RESIZE_CV2_LINEAR = 2

    def __init__(self, dpt, fx, fy, importer=None):
        """
        Constructor
        :param dpt: depth image
        :param fx: camera focal lenght
        :param fy: camera focal lenght
        """
        self.dpt = dpt
        self.maxDepth = min(1500, dpt.max())
        self.minDepth = max(10, dpt.min())
        # set values out of range to 0
        self.dpt[self.dpt > self.maxDepth] = 0.
        self.dpt[self.dpt < self.minDepth] = 0.
        # camera settings
        self.fx = fx
        self.fy = fy
        self.importer = importer
        # depth resize method
        self.resizeMethod = self.RESIZE_CV2_NN

    def calculateCoM(self, dpt):
        """
        Calculate the center of mass
        :param dpt: depth image
        :return: (x,y,z) center of mass
        """

        dc = dpt.copy()
        dc[dc < self.minDepth] = 0
        dc[dc > self.maxDepth] = 0
        cc = ndimage.measurements.center_of_mass(dc > 0)
        num = numpy.count_nonzero(dc)
        com = numpy.array((cc[1]*num, cc[0]*num, dc.sum()), numpy.float)

        if num == 0:
            return numpy.array((0, 0, 0), numpy.float)
        else:
            return com/num

    def checkImage(self, tol):
        """
        Check if there is some content in the image
        :param tol: tolerance
        :return:True if image is contentful, otherwise false
        """
        # print numpy.std(self.dpt)
        if numpy.std(self.dpt) < tol:
            return False
        else:
            return True

    def getNDValue(self):
        """
        Get value of not defined depth value distances
        :return:value of not defined depth value
        """
        if self.dpt[self.dpt < self.minDepth].shape[0] > self.dpt[self.dpt > self.maxDepth].shape[0]:
            return stats.mode(self.dpt[self.dpt < self.minDepth])[0][0]
        else:
            return stats.mode(self.dpt[self.dpt > self.maxDepth])[0][0]

    @staticmethod
    def bilinearResize(src, dsize, ndValue):
        """
        Bilinear resizing with sparing out not defined parts of the depth map
        :param src: source depth map
        :param dsize: new size of resized depth map
        :param ndValue: value of not defined depth
        :return:resized depth map
        """

        dst = numpy.zeros((dsize[1], dsize[0]), dtype=numpy.float32)

        x_ratio = float(src.shape[1] - 1) / dst.shape[1]
        y_ratio = float(src.shape[0] - 1) / dst.shape[0]
        for row in range(dst.shape[0]):
            y = int(row * y_ratio)
            y_diff = (row * y_ratio) - y  # distance of the nearest pixel(y axis)
            y_diff_2 = 1 - y_diff
            for col in range(dst.shape[1]):
                x = int(col * x_ratio)
                x_diff = (col * x_ratio) - x  # distance of the nearest pixel(x axis)
                x_diff_2 = 1 - x_diff
                y2_cross_x2 = y_diff_2 * x_diff_2
                y2_cross_x = y_diff_2 * x_diff
                y_cross_x2 = y_diff * x_diff_2
                y_cross_x = y_diff * x_diff

                # mathematically impossible, but just to be sure...
                if(x+1 >= src.shape[1]) | (y+1 >= src.shape[0]):
                    raise UserWarning("Shape mismatch")

                # set value to ND if there are more than two values ND
                numND = int(src[y, x] == ndValue) + int(src[y, x + 1] == ndValue) + int(src[y + 1, x] == ndValue) + int(
                    src[y + 1, x + 1] == ndValue)
                if numND > 2:
                    dst[row, col] = ndValue
                    continue
                # print y2_cross_x2, y2_cross_x, y_cross_x2, y_cross_x
                # interpolate only over known values, switch to linear interpolation
                if src[y, x] == ndValue:
                    y2_cross_x2 = 0.
                    y2_cross_x = 1. - y_cross_x - y_cross_x2
                if src[y, x + 1] == ndValue:
                    y2_cross_x = 0.
                    if y2_cross_x2 != 0.:
                        y2_cross_x2 = 1. - y_cross_x - y_cross_x2
                if src[y + 1, x] == ndValue:
                    y_cross_x2 = 0.
                    y_cross_x = 1. - y2_cross_x - y2_cross_x2
                if src[y + 1, x + 1] == ndValue:
                    y_cross_x = 0.
                    if y_cross_x2 != 0.:
                        y_cross_x2 = 1. - y2_cross_x - y2_cross_x2

                # print src[y, x], src[y, x+1],src[y+1, x],src[y+1, x+1]
                # normalize weights
                if not ((y2_cross_x2 == 0.) & (y2_cross_x == 0.) & (y_cross_x2 == 0.) & (y_cross_x == 0.)):
                    sc = 1. / (y_cross_x + y_cross_x2 + y2_cross_x + y2_cross_x2)
                    y2_cross_x2 *= sc
                    y2_cross_x *= sc
                    y_cross_x2 *= sc
                    y_cross_x *= sc
                # print y2_cross_x2, y2_cross_x, y_cross_x2, y_cross_x

                if (y2_cross_x2 == 0.) & (y2_cross_x == 0.) & (y_cross_x2 == 0.) & (y_cross_x == 0.):
                    dst[row, col] = ndValue
                else:
                    dst[row, col] = y2_cross_x2 * src[y, x] + y2_cross_x * src[y, x + 1] + y_cross_x2 * src[
                        y + 1, x] + y_cross_x * src[y + 1, x + 1]

        return dst

    def comToBounds(self, com, size):
        """
        Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: xstart, xend, ystart, yend, zstart, zend
        """
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(numpy.floor((com[0] * com[2] / self.fx - size[0] / 2.) / com[2]*self.fx))
        xend = int(numpy.floor((com[0] * com[2] / self.fx + size[0] / 2.) / com[2]*self.fx))
        ystart = int(numpy.floor((com[1] * com[2] / self.fy - size[1] / 2.) / com[2]*self.fy))
        yend = int(numpy.floor((com[1] * com[2] / self.fy + size[1] / 2.) / com[2]*self.fy))
        return xstart, xend, ystart, yend, zstart, zend

    def getCrop(self, dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z=True):
        """
        Crop patch from image
        :param dpt: depth image to crop from
        :param xstart: start x
        :param xend: end x
        :param ystart: start y
        :param yend: end y
        :param zstart: start z
        :param zend: end z
        :param thresh_z: threshold z values
        :return: cropped image
        """
        if len(dpt.shape) == 2:
            cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1])].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                           abs(yend)-min(yend, dpt.shape[0])),
                                          (abs(xstart)-max(xstart, 0),
                                           abs(xend)-min(xend, dpt.shape[1]))), mode='constant', constant_values=0)
        elif len(dpt.shape) == 3:
            cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1]), :].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                           abs(yend)-min(yend, dpt.shape[0])),
                                          (abs(xstart)-max(xstart, 0),
                                           abs(xend)-min(xend, dpt.shape[1])),
                                          (0, 0)), mode='constant', constant_values=0)
        else:
            raise NotImplementedError()

        if thresh_z is True:
            msk1 = numpy.bitwise_and(cropped < zstart, cropped != 0)
            msk2 = numpy.bitwise_and(cropped > zend, cropped != 0)
            cropped[msk1] = zstart
            cropped[msk2] = 0.  # backface is at 0, it is set later
        return cropped

    def resizeCrop(self, crop, sz):
        """
        Resize cropped image
        :param crop: crop
        :param sz: size
        :return: resized image
        """
        if self.resizeMethod == self.RESIZE_CV2_NN:
            rz = cv2.resize(crop, sz, interpolation=cv2.INTER_NEAREST)
        elif self.resizeMethod == self.RESIZE_BILINEAR:
            rz = self.bilinearResize(crop, sz, self.getNDValue())
        elif self.resizeMethod == self.RESIZE_CV2_LINEAR:
            rz = cv2.resize(crop, sz, interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError("Unknown resize method!")
        return rz

    def applyCrop3D(self, dpt, com, size, dsize, thresh_z=True, background=None):

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

        # crop patch from source
        cropped = self.getCrop(dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z)

        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            sz = (wb * dsize[1] / hb, dsize[1])

        # depth resize
        rz = self.resizeCrop(cropped, sz)

        if background is None:
            background = self.getNDValue()  # use background as filler
        ret = numpy.ones(dsize, numpy.float32) * background
        xstart = int(numpy.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(numpy.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz

        return ret

    def cropArea3D(self, com=None, size=(250, 250, 250), dsize=(128, 128), docom=False):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        # print com, self.importer.jointImgTo3D(com)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        if com is None:
            com = self.calculateCoM(self.dpt)

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

        # crop patch from source
        cropped = self.getCrop(self.dpt, xstart, xend, ystart, yend, zstart, zend)
        # ax.plot(com[0],com[1],marker='.')

        #############
        # for simulating COM within cube
        if docom is True:
            com = self.calculateCoM(cropped)
            if numpy.allclose(com, 0.):
                com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
            com[0] += xstart
            com[1] += ystart

            # calculate boundaries
            xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

            # crop patch from source
            cropped = self.getCrop(self.dpt, xstart, xend, ystart, yend, zstart, zend)
        # ax.plot(com[0],com[1],marker='x')
        #############

        wb = (xend - xstart)
        hb = (yend - ystart)
        trans = numpy.asmatrix(numpy.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            sz = (wb * dsize[1] / hb, dsize[1])

        # print com, sz, cropped.shape, xstart, xend, ystart, yend, hb, wb, zstart, zend
        if cropped.shape[0] > cropped.shape[1]:
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[1] / float(cropped.shape[0]))
        else:
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[0] / float(cropped.shape[1]))
        scale[2, 2] = 1

        # depth resize
        rz = self.resizeCrop(cropped, sz)

        # pylab.imshow(rz); pylab.gray();t=transformPoint2D(com,scale*trans);pylab.scatter(t[0],t[1]); pylab.show()
        ret = numpy.ones(dsize, numpy.float32) * self.getNDValue()  # use background as filler
        xstart = int(numpy.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(numpy.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape
        off = numpy.asmatrix(numpy.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        # fig = plt.figure()
        # ax = fig.add_subplot(131)
        # ax.imshow(cropped, cmap=matplotlib.cm.jet)
        # ax = fig.add_subplot(132)
        # ax.imshow(rz, cmap=matplotlib.cm.jet)
        # ax = fig.add_subplot(133)
        # ax.imshow(ret, cmap=matplotlib.cm.jet)
        # plt.show(block=False)

        # print trans,scale,off,off*scale*trans
        return ret, off * scale * trans, com

    def detectFromCache(self, cache_name, name, cache_str=''):
        """
        Load hand detection from cached file
        :param cache_name: cache filename
        :param name: hand filename
        :return: center of hand
        """

        com = (0, 0, 0)

        if cache_str != '':
            count = 0
            for line in cache_str:
                part = line.strip().split(' ')
                if isinstance(name, basestring) and os.path.normpath(part[0]).split(os.path.sep)[-1] == os.path.normpath(name).split(os.path.sep)[-1] and os.path.normpath(part[0]).split(os.path.sep)[-2] == os.path.normpath(name).split(os.path.sep)[-2]:
                    assert len(part) == 4
                    com = (float(part[1]), float(part[2]), float(part[3]))
                    break
                if isinstance(name, int) and count == name:
                    assert len(part) == 3
                    com = (float(part[0]), float(part[1]), float(part[2]))
                    break
                count += 1
        elif os.path.isfile(cache_name):
            with open(cache_name, 'r') as inputfile:
                count = 0
                for line in inputfile:
                    part = line.strip().split(' ')
                    if isinstance(name, basestring) and os.path.normpath(part[0]).split(os.path.sep)[-1] == os.path.normpath(name).split(os.path.sep)[-1] and os.path.normpath(part[0]).split(os.path.sep)[-2] == os.path.normpath(name).split(os.path.sep)[-2]:
                        assert len(part) == 4
                        com = (float(part[1]), float(part[2]), float(part[3]))
                        break
                    if isinstance(name, int) and count == name:
                        assert len(part) == 3
                        com = (float(part[0]), float(part[1]), float(part[2]))
                        break
                    count += 1

        if numpy.allclose(com, 0.):
            print "WARNING: File {} not found in {}!".format(name, cache_name)

        return com

