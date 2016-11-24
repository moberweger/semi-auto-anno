"""Provides class for evaluating hand pose accuracy.

HandposeEvaluation provides interface for evaluating the hand pose accuracy.
BlenderHandposeEvaluation is a specific instance for different dataset.

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
from data.importers import DepthImporter, Blender2Importer
from data.transformations import transformPoint2D
import progressbar as pb

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class HandposeEvaluation(object):
    """
    Different evaluation metrics for handpose, L2 distance used
    """

    def __init__(self, gt, joints, dolegend=True, linewidth=1):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        if not (isinstance(gt, numpy.ndarray) or isinstance(gt, list)) or not (
                isinstance(joints, list) or isinstance(joints, numpy.ndarray)):
            raise ValueError("Params must be list or ndarray")

        if len(gt) != len(joints):
            print("Error: groundtruth has {} elements, eval data has {}".format(len(gt), len(joints)))
            raise ValueError("Params must be the same size")

        if len(gt) == len(joints) == 0:
            print("Error: groundtruth has {} elements, eval data has {}".format(len(gt), len(joints)))
            raise ValueError("Params must be of non-zero size")

        if gt[0].shape != joints[0].shape:
            print("Error: groundtruth has {} dims, eval data has {}".format(gt[0].shape, joints[0].shape))
            raise ValueError("Params must be of same dimensionality")

        self.gt = numpy.asarray(gt)
        self.joints = numpy.asarray(joints)
        assert (self.gt.shape == self.joints.shape)

        self.colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'brown', 'gray', 'indigo', 'pink',
                       'lightgreen', 'darkorange', 'peru', 'steelblue', 'turquoise']
        self.linestyles = ['-']  # , '--', '-.', ':', '-', '--', '-.', ':']
        self.linewidth = linewidth
        self.dolegend = dolegend

        self.subfolder = './eval/'
        self.visiblemask = numpy.ones((self.gt.shape[0], self.gt.shape[1], 3))

        self.jointNames = None
        self.jointConnections = []
        self.jointConnectionColors = []
        self.plotMaxJointDist = 80
        self.plotMeanJointDist = 80
        self.plotMedianJointDist = 80
        self.VTKviewport = [0, 0, 0, 0, 0]

    def maskVisibility(self, msk):
        """
        Checks the visibility of the joints and masks out the occluded ones
        :param msk: visibility mask, 0 if not visible, 1 if visible
        :return: None
        """

        assert msk.shape[0] == self.gt.shape[0] and msk.shape[1] == self.gt.shape[1], "{}, {}".format(msk.shape, self.gt.shape)

        if len(msk.shape) == 2:
            msk = numpy.concatenate((msk[:, :, None], msk[:, :, None], msk[:, :, None]), axis=2)

        self.visiblemask = numpy.ones((self.gt.shape[0], self.gt.shape[1], 3))
        # remove not visible ones
        self.visiblemask[msk == 0] = numpy.nan
        self.gt = numpy.multiply(self.gt, self.visiblemask)
        self.joints = numpy.multiply(self.joints, self.visiblemask)

    def getJointNumFramesVisible(self, jointID):
        """
        Get number of frames in which joint is visible
        :param jointID: joint ID
        :return: number of frames
        """

        return numpy.nansum(self.gt[:, jointID, :]) / self.gt.shape[2]  # 3D

    def getMeanError(self):
        """
        get average error over all joints, averaged over sequence
        :return: mean error
        """
        return numpy.nanmean(numpy.nanmean(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1))

    def getStdError(self):
        """
        get standard deviation of error over all joints, averaged over sequence
        :return: standard deviation of error
        """
        return numpy.nanmean(numpy.nanstd(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1))

    def getMedianError(self):
        """
        get median error over all joints
        :return: median error
        """

        errs = numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2))
        return numpy.median(errs[numpy.isfinite(errs)])

    def getMaxError(self):
        """
        get max error over all joints
        :return: maximum error
        """

        return numpy.nanmax(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)))

    def getJointMeanError(self, jointID):
        """
        get error of one joint, averaged over sequence
        :param jointID: joint ID
        :return: mean joint error
        """

        return numpy.nanmean(numpy.sqrt(numpy.square(self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))

    def getJointStdError(self, jointID):
        """
        get standard deviation of one joint, averaged over sequence
        :param jointID: joint ID
        :return: standard deviation of joint error
        """

        return numpy.nanstd(numpy.sqrt(numpy.square(self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))

    def getJointDiffOverSeq(self, jointID):
        """
        get error of one joint for each image of sequence
        :param jointID: joint ID
        :return: joint error
        """

        return self.gt[:, jointID, :] - self.joints[:, jointID, :]

    def getJointMaxError(self, jointID):
        """
        get maximum error of one joint
        :param jointID: joint ID
        :return: maximum joint error
        """

        return numpy.nanmax(numpy.sqrt(numpy.square(self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))

    def getNumFramesWithinMaxDist(self, dist):
        """
        calculate the number of frames where the maximum difference of a joint is within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.nanmax(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1) <= dist).sum()

    def plotFramesWithinMax(self, basename, methodName='Our method', baseline=None):
        """
        plot and save plot for fraction of frames within max distance
        :param basename: file basename
        :param methodName: our method name
        :param baseline: list of baselines as tuple (Name,evaluation object)
        :return: None
        """

        if baseline is not None:
            for bs in baseline:
                if not (isinstance(bs[1], self.__class__)):
                    raise TypeError('baseline must be of type {} but {} provided'.format(self.__class__.__name__,
                                                                                         bs[1].__class__.__name__))

        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([self.getNumFramesWithinMaxDist(j) / float(self.joints.shape[0]) * 100. for j in
                 range(0, self.plotMaxJointDist)], label=methodName, c=self.colors[0], linestyle=self.linestyles[0],
                linewidth=self.linewidth)
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.plot([bs[1].getNumFramesWithinMaxDist(j) / float(self.joints.shape[0]) * 100. for j in
                         range(0, self.plotMaxJointDist)], label=bs[0], c=self.colors[bs_idx % len(self.colors)],
                        linestyle=self.linestyles[bs_idx % len(self.linestyles)], linewidth=self.linewidth)
                bs_idx += 1
        plt.xlabel('Distance threshold / mm')
        plt.ylabel('Fraction of frames within distance / %')
        plt.ylim([0.0, 100.0])
        ax.grid(True)
        if self.dolegend:
            # Put a legend below current axis
            handles, labels = ax.get_legend_handles_labels()
            # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)  # ncol=2, prop={'size': 14})
            bbea = (lgd,)
        else:
            bbea = None
        plt.show(block=False)
        fig.savefig('{}/{}_frameswithin.pdf'.format(self.subfolder, basename), bbox_extra_artists=bbea,
                    bbox_inches='tight')
        plt.close(fig)

    def plotJointMeanError(self, basename, methodName='Our method', baseline=None):
        """
        plot and save plot for mean error for each joint
        :param basename: file basename
        :param methodName: our method name
        :param baseline: list of baselines as tuple (Name,evaluation object)
        :return: None
        """

        if baseline is not None:
            for bs in baseline:
                if not (isinstance(bs[1], self.__class__)):
                    raise TypeError('baseline must be of type {} but {} provided'.format(self.__class__.__name__,
                                                                                         bs[1].__class__.__name__))

        import matplotlib.pyplot as plt

        ind = numpy.arange(self.joints.shape[1]+1)  # the x locations for the groups, +1 for mean
        if baseline is not None:
            width = (1 - 0.33) / (1. + len(baseline))  # the width of the bars
        else:
            width = 0.67
        fig, ax = plt.subplots()
        mean = [self.getJointMeanError(j) for j in range(self.joints.shape[1])]
        mean.append(self.getMeanError())
        std = [self.getJointStdError(j) for j in range(self.joints.shape[1])]
        std.append(self.getStdError())
        ax.bar(ind, numpy.array(mean), width, label=methodName, color=self.colors[0])  # , yerr=std)
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                mean = [bs[1].getJointMeanError(j) for j in range(self.joints.shape[1])]
                mean.append(bs[1].getMeanError())
                std = [bs[1].getJointStdError(j) for j in range(self.joints.shape[1])]
                std.append(bs[1].getStdError())
                ax.bar(ind + width * float(bs_idx), numpy.array(mean), width,
                       label=bs[0], color=self.colors[bs_idx % len(self.colors)])  # , yerr=std)
                bs_idx += 1
        ax.set_xticks(ind + width)
        ll = list(self.jointNames)
        ll.append('Avg')
        label = tuple(ll)
        ax.set_xticklabels(label)
        plt.ylabel('Mean error of joint / mm')
        # plt.ylim([0.0,50.0])
        if self.dolegend:
            # Put a legend below current axis
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
            bbea = (lgd,)
        else:
            bbea = None
        plt.show(block=False)
        fig.savefig('{}/{}_joint_mean.pdf'.format(self.subfolder, basename), bbox_extra_artists=bbea,
                    bbox_inches='tight')
        plt.close(fig)

    def plotJointMaxError(self, basename, methodName='Our method', baseline=None):
        """
        plot and save plot for maximum error for each joint
        :param basename: file basename
        :param methodName: our method name
        :param baseline: list of baselines as tuple (Name,evaluation object)
        :return: None
        """

        if baseline is not None:
            for bs in baseline:
                if not (isinstance(bs[1], self.__class__)):
                    raise TypeError('baseline must be of type {} but {} provided'.format(self.__class__.__name__,
                                                                                         bs[1].__class__.__name__))

        import matplotlib.pyplot as plt

        ind = numpy.arange(self.joints.shape[1])  # the x locations for the groups
        if baseline is not None:
            width = (1 - 0.33) / (1. + len(baseline))  # the width of the bars
        else:
            width = 0.67
        fig, ax = plt.subplots()
        ax.bar(ind, numpy.array([self.getJointMaxError(j) for j in range(self.joints.shape[1])]), width,
               label=methodName, color=self.colors[0])
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.bar(ind + width * float(bs_idx),
                       numpy.array([bs[1].getJointMaxError(j) for j in range(self.joints.shape[1])]), width,
                       label=bs[0], color=self.colors[bs_idx % len(self.colors)])
                bs_idx += 1
        ax.set_xticks(ind + width)
        ax.set_xticklabels(self.jointNames)
        plt.ylabel('Maximum error of joint / mm')
        plt.ylim([0.0, 200.0])
        if self.dolegend:
            # Put a legend below current axis
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
            bbea = (lgd,)
        else:
            bbea = None
        plt.show(block=False)
        fig.savefig('{}/{}_joint_max.pdf'.format(self.subfolder, basename), bbox_extra_artists=bbea,
                    bbox_inches='tight')
        plt.close(fig)

    def plotEvaluation(self, basename, methodName='Our method', baseline=None):
        """
        plot and save standard evaluation plots
        :param basename: file basename
        :param methodName: our method name
        :param baseline: list of baselines as tuple (Name,evaluation object)
        :return: None
        """

        if baseline is not None:
            for bs in baseline:
                if not (isinstance(bs[1], self.__class__)):
                    raise TypeError('baseline must be of type {} but {} provided'.format(self.__class__.__name__,
                                                                                         bs[1].__class__.__name__))

        # plot number of frames within max distance
        self.plotFramesWithinMax(basename, methodName, baseline)

        # plot mean error for each joint
        self.plotJointMeanError(basename, methodName, baseline)

        # plot maximum error for each joint
        self.plotJointMaxError(basename, methodName, baseline)

    def plotResult(self, dpt, gtcrop, joint, name=None, showGT=True, niceColors=False, showJoints=True, showDepth=True,
                   visibility=None, highlight=None, upsample=4., annoscale=1):
        """
        Show the annotated depth image
        :param dpt: depth image to show
        :param gtcrop: cropped 2D coordinates
        :param joint: joint data
        :param name: name of file to save, if None return image
        :param showGT: show groundtruth annotation
        :param niceColors: plot nice gradient colors for each joint
        :param visibility: plot different markers for visible/non-visible joints, visible are marked as 1, non-visible 0
        :param highlight: highlight different joints, set as 1 to highlight, otherwise 0
        :return: None, or image if name = None
        """

        # plot depth image with annotations
        if showDepth:
            imgcopy = dpt.copy()
            # display hack to hide nd depth
            msk = imgcopy > 0
            msk2 = imgcopy == 0
            min = imgcopy[msk].min()
            max = imgcopy[msk].max()
            imgcopy = (imgcopy - min) / (max - min) * 255.
            imgcopy[msk2] = 255.
        else:
            # same view as with image
            imgcopy = numpy.ones_like(dpt)*255.

        # resize image
        if len(imgcopy.shape) == 2:
            imgcopy = cv2.cvtColor(imgcopy.astype('uint8'), cv2.COLOR_GRAY2BGR)
        elif len(imgcopy.shape) == 3:
            imgcopy = imgcopy.astype('uint8')
        else:
            raise NotImplementedError("")

        if not numpy.allclose(upsample, 1):
            imgcopy = cv2.resize(imgcopy, dsize=None, fx=upsample, fy=upsample, interpolation=cv2.INTER_LINEAR)

        joint = joint.copy()
        for i in range(joint.shape[0]):
            joint[i, 0:2] -= numpy.asarray([dpt.shape[0]//2, dpt.shape[1]//2])
            joint[i, 0:2] *= upsample
            joint[i, 0:2] += numpy.asarray([imgcopy.shape[0]//2, imgcopy.shape[1]//2])

        gtcrop = gtcrop.copy()
        for i in range(gtcrop.shape[0]):
            gtcrop[i, 0:2] -= numpy.asarray([dpt.shape[0]//2, dpt.shape[1]//2])
            gtcrop[i, 0:2] *= upsample
            gtcrop[i, 0:2] += numpy.asarray([imgcopy.shape[0]//2, imgcopy.shape[1]//2])

        # use child class plots
        if showJoints:
            self.plotJoints(imgcopy, joint, visibility=visibility, highlight=highlight, annoscale=annoscale,
                            color=((0, 0, 255) if niceColors is False else 'nice'),
                            jcolor=((0, 0, 255) if niceColors is False else 'nice'))  # ours
        if showGT:
            if showJoints and showGT and (niceColors is True):
                cc = 'gray'
                jc = 'gray'
            elif niceColors is False:
                cc = (255, 0, 0)
                jc = (255, 0, 0)
            else:
                cc = 'nice'
                jc = 'nice'

            self.plotJoints(imgcopy, gtcrop, visibility=visibility, highlight=highlight, annoscale=annoscale,
                            color=cc, jcolor=jc)  # groundtruth
        if name is not None:
            cv2.imwrite('{}/annotated_{}.png'.format(self.subfolder, name), imgcopy)
        else:
            import matplotlib.pyplot as plt

            if plt.matplotlib.get_backend() == 'agg':
                return imgcopy
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.imshow(imgcopy)
                plt.tight_layout(pad=0)
                plt.show(block=False)
                return imgcopy

    def plotJoints(self, ax, joint, visibility=None, highlight=None, color='nice', jcolor=None, annoscale=1):
        """
        Plot connected joints
        :param ax: axis to plot on
        :param visibility: plot different markers for visible/non-visible joints, visible are marked as 1, non-visible 0
        :param highlight: highlight different joints, set as 1 to highlight, otherwise 0
        :param joint: joints to connect
        :param color: line color
        """

        # all are visible by default
        if visibility is None:
            visibility = numpy.ones((joint.shape[0],))
        else:
            assert visibility.shape[0] == joint.shape[0]

        # all are non-highlighted by default
        if highlight is None:
            highlight = numpy.zeros((joint.shape[0],))
        else:
            assert highlight.shape[0] == joint.shape[0]

        for i in range(len(self.jointConnections)):
            if isinstance(ax, numpy.ndarray):
                if color == 'nice':
                    lc = tuple((self.jointConnectionColors[i]*255.).astype(int))
                elif color == 'gray':
                    lc = tuple((self.rgb_to_gray(self.jointConnectionColors[i])*255.).astype(int))
                else:
                    lc = color
                cv2.line(ax, (int(numpy.rint(joint[self.jointConnections[i][0], 0])),
                              int(numpy.rint(joint[self.jointConnections[i][0], 1]))),
                         (int(numpy.rint(joint[self.jointConnections[i][1], 0])),
                          int(numpy.rint(joint[self.jointConnections[i][1], 1]))),
                         lc, thickness=3*annoscale, lineType=cv2.CV_AA)
            else:
                if color == 'nice':
                    lc = self.jointConnectionColors[i]
                elif color == 'gray':
                    lc = self.rgb_to_gray(self.jointConnectionColors[i])
                else:
                    lc = color
                ax.plot(numpy.hstack((joint[self.jointConnections[i][0], 0], joint[self.jointConnections[i][1], 0])),
                        numpy.hstack((joint[self.jointConnections[i][0], 1], joint[self.jointConnections[i][1], 1])),
                        c=lc, linewidth=3.0*annoscale)
        for i in range(joint.shape[0]):
            if isinstance(ax, numpy.ndarray):
                if numpy.allclose(highlight[i], 1.):
                    cv2.circle(ax, (int(numpy.rint(joint[i, 0])), int(numpy.rint(joint[i, 1]))), 10*annoscale,
                               (0, 0, 255), thickness=-1, lineType=cv2.CV_AA)

                if jcolor == 'nice':
                    jc = tuple((self.jointColors[i]*255.).astype(int))
                elif jcolor == 'gray':
                    jc = tuple((self.rgb_to_gray(self.jointColors[i])*255.).astype(int))
                else:
                    jc = jcolor
                if numpy.allclose(visibility[i], 1.):
                    cv2.circle(ax, (int(numpy.rint(joint[i, 0])), int(numpy.rint(joint[i, 1]))), 6*annoscale,
                               jc, thickness=-1, lineType=cv2.CV_AA)
                else:
                    triangle = numpy.array([[-4, -4], [-4, 4], [4, 0]])*annoscale + numpy.asarray((joint[i, 0], joint[i, 1]))
                    cv2.fillConvexPoly(ax, numpy.rint(triangle).astype(int),
                                       jc, lineType=cv2.CV_AA)
            else:
                if numpy.allclose(highlight[i], 1.):
                    import matplotlib
                    ax.add_artist(matplotlib.pyplot.Circle((joint[i, 0], joint[i, 1]), 10*annoscale, color='r'))

                if jcolor == 'nice':
                    jc = self.jointColors[i]
                elif jcolor == 'gray':
                    jc = self.rgb_to_gray(self.jointColors[i])
                else:
                    jc = jcolor

                ax.scatter(joint[i, 0], joint[i, 1], marker='o' if numpy.allclose(visibility[i], 1.) else 'v', s=100,
                           c=jc)

    def plotResult3D(self, dpt, T, gt3Dorig, joint3D, visibility=None, filename=None, showGT=True, showPC=True,
                     niceColors=False):
        """
        Plot 3D point cloud
        :param dpt: depth image
        :param T: 2D image transformation
        :param gt3Dorig: groundtruth 3D pose
        :param joint3D: 3D joint data
        :param visibility: plot different markers for visible/non-visible joints, visible are marked as 1, non-visible 0
        :param filename: name of file to save, if None return image
        :param showGT: show groundtruth annotation
        :param showPC: show point cloud
        :return: None, or image if filename=None
        """

        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        from util.vtkpointcloud import VtkPointCloud
        import matplotlib.pyplot as plt

        def close_window(iren):
            render_window = iren.GetRenderWindow()
            render_window.Finalize()
            iren.TerminateApp()

        def key_pressed_callback(obj, event):
            key = obj.GetKeySym()
            iren = obj
            render_window = iren.GetRenderWindow()
            if key == "s":
                file_name = self.subfolder + str(numpy.random.randint(0, 100)).zfill(5) + ".png"
                image = vtk.vtkWindowToImageFilter()
                image.SetInput(render_window)
                png_writer = vtk.vtkPNGWriter()
                png_writer.SetInputConnection(image.GetOutputPort())
                png_writer.SetFileName(file_name)
                render_window.Render()
                png_writer.Write()
            elif key == "c":
                camera = renderer.GetActiveCamera()
                print "Camera settings:"
                print "  * position:        %s" % (camera.GetPosition(),)
                print "  * focal point:     %s" % (camera.GetFocalPoint(),)
                print "  * up vector:       %s" % (camera.GetViewUp(),)

        class vtkTimerCallback():
            def __init__(self):
                pass

            def execute(self, obj, event):
                if plt.matplotlib.get_backend() == 'agg':
                    iren = obj
                    render_window = iren.GetRenderWindow()
                    render_window.Finalize()
                    iren.TerminateApp()
                    del render_window, iren

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1.0, 1.0, 1.0)

        if showPC is True:
            pointCloud = VtkPointCloud()

            pcl = self.getPCL(dpt, T)

            for k in xrange(pcl.shape[0]):
                point = pcl[k]
                pointCloud.addPoint(point)

            renderer.AddActor(pointCloud.vtkActor)
            renderer.ResetCamera()

        self.vtkPlotHand(renderer, joint3D, visibility, 'nice' if niceColors is True else (1, 0, 0))
        if showGT:
            self.vtkPlotHand(renderer, gt3Dorig, visibility, 'nice' if niceColors is True else (0, 0, 1))

        # setup camera position
        camera = renderer.GetActiveCamera()
        camera.Pitch(self.VTKviewport[0])
        camera.Yaw(self.VTKviewport[1])
        camera.Roll(self.VTKviewport[2])
        camera.Azimuth(self.VTKviewport[3])
        camera.Elevation(self.VTKviewport[4])

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderWindowInteractor.AddObserver("KeyPressEvent", key_pressed_callback)

        if showPC is False:
            renderer.ResetCamera()
            # setup camera position
            camera = renderer.GetActiveCamera()
            camera.Pitch(self.VTKviewport[0])
            camera.Yaw(self.VTKviewport[1])
            camera.Roll(self.VTKviewport[2])
            camera.Azimuth(self.VTKviewport[3])
            camera.Elevation(self.VTKviewport[4])

        # Begin Interaction
        renderWindow.Render()
        renderWindow.SetWindowName("XYZ Data Viewer")

        # Sign up to receive TimerEvent
        cb = vtkTimerCallback()
        cb.actor = renderer.GetActors().GetLastActor()
        renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
        timerId = renderWindowInteractor.CreateRepeatingTimer(10)

        renderWindowInteractor.Start()

        if filename is not None:
            im = vtk.vtkWindowToImageFilter()
            writer = vtk.vtkPNGWriter()
            im.SetInput(renderWindow)
            im.Update()
            writer.SetInputConnection(im.GetOutputPort())
            writer.SetFileName('{}/{}.png'.format(self.subfolder, filename))
            writer.Write()
            close_window(renderWindowInteractor)
            del renderWindow, renderWindowInteractor
        else:
            im = vtk.vtkWindowToImageFilter()
            im.SetInput(renderWindow)
            im.Update()
            vtk_image = im.GetOutput()
            height, width, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            close_window(renderWindowInteractor)
            del renderWindow, renderWindowInteractor
            return vtk_to_numpy(vtk_array).reshape(height, width, components)

    def plotResult3D_OS(self, dpt, T, gt3Dorig, joint3D, visibility=None, filename=None, showGT=True, showPC=True,
                        niceColors=False, width=300, height=300):
        """
        Plot 3D point cloud
        :param dpt: depth image
        :param T: 2D image transformation
        :param gt3Dorig: groundtruth 3D pose
        :param joint3D: 3D joint data
        :param visibility: plot different markers for visible/non-visible joints, visible are marked as 1, non-visible 0
        :param filename: name of file to save, if None return image
        :param showGT: show groundtruth annotation
        :param showPC: show point cloud
        :param width: width of window
        :param height: height of window
        :return: None, or image if filename=None
        """

        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        from util.vtkpointcloud import VtkPointCloud

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1.0, 1.0, 1.0)

        if showPC is True:
            pointCloud = VtkPointCloud()

            pcl = self.getPCL(dpt, T)

            for k in xrange(pcl.shape[0]):
                point = pcl[k]
                pointCloud.addPoint(point)

            renderer.AddActor(pointCloud.vtkActor)
            renderer.ResetCamera()

        self.vtkPlotHand(renderer, joint3D, visibility, 'nice' if niceColors is True else (1, 0, 0))
        if showGT:
            self.vtkPlotHand(renderer, gt3Dorig, visibility, 'nice' if niceColors is True else (0, 0, 1))

        # setup camera position
        camera = renderer.GetActiveCamera()
        camera.Pitch(self.VTKviewport[0])
        camera.Yaw(self.VTKviewport[1])
        camera.Roll(self.VTKviewport[2])
        camera.Azimuth(self.VTKviewport[3])
        camera.Elevation(self.VTKviewport[4])

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetOffScreenRendering(1)
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(width, height)

        if showPC is False:
            renderer.ResetCamera()
            # setup camera position
            camera = renderer.GetActiveCamera()
            camera.Pitch(self.VTKviewport[0])
            camera.Yaw(self.VTKviewport[1])
            camera.Roll(self.VTKviewport[2])
            camera.Azimuth(self.VTKviewport[3])
            camera.Elevation(self.VTKviewport[4])

        # Render window
        renderWindow.Render()

        if filename is not None:
            im = vtk.vtkWindowToImageFilter()
            writer = vtk.vtkPNGWriter()
            im.SetInput(renderWindow)
            im.Update()
            writer.SetInputConnection(im.GetOutputPort())
            writer.SetFileName('{}/{}.png'.format(self.subfolder, filename))
            writer.Write()
        else:
            im = vtk.vtkWindowToImageFilter()
            im.SetInput(renderWindow)
            im.Update()
            vtk_image = im.GetOutput()
            height, width, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            return vtk_to_numpy(vtk_array).reshape(height, width, components)

    def vtkPlotHand(self, renderer, joint3D, visibility=None, colors=(1, 0, 0)):
        """
        Plot hand in vtk renderer, as a stick and ball model
        :param renderer: vtk renderer instance
        :param joint3D: 3D joint locations
        :param visibility: visibility information, default all visible
        :param colors: colors of joints or 'nice'
        :return: None
        """

        import vtk

        # all are visible by default
        if visibility is None:
            visibility = numpy.ones((joint3D.shape[0],))
        else:
            assert visibility.shape[0] == joint3D.shape[0]

        for i in range(joint3D.shape[0]):
            if numpy.allclose(visibility[i], 1.):
                # create source
                source = vtk.vtkSphereSource()
                source.SetCenter(joint3D[i, 0], joint3D[i, 1], joint3D[i, 2])
                source.SetRadius(5.0)
                # mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())
            else:
                # create source
                source = vtk.vtkCubeSource()
                source.SetCenter(joint3D[i, 0], joint3D[i, 1], joint3D[i, 2])
                source.SetXLength(5.0)
                source.SetYLength(5.0)
                source.SetZLength(5.0)
                # mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())

            # actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # color actor
            if colors == 'nice':
                actor.GetProperty().SetColor(self.jointColors[i][0],
                                             self.jointColors[i][1],
                                             self.jointColors[i][2])
            else:
                actor.GetProperty().SetColor(colors[0], colors[1], colors[2])

            # assign actor to the renderer
            renderer.AddActor(actor)

        if joint3D.shape[0] >= numpy.max(self.jointConnections):
            for i in range(len(self.jointConnections)):
                # create source
                source = vtk.vtkLineSource()
                source.SetPoint1(joint3D[self.jointConnections[i][0], 0], joint3D[self.jointConnections[i][0], 1], joint3D[self.jointConnections[i][0], 2])
                source.SetPoint2(joint3D[self.jointConnections[i][1], 0], joint3D[self.jointConnections[i][1], 1], joint3D[self.jointConnections[i][1], 2])

                # mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())

                # actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # color actor
                if colors == 'nice':
                    actor.GetProperty().SetColor(self.jointConnectionColors[i][0], self.jointConnectionColors[i][1], self.jointConnectionColors[i][2])
                else:
                    actor.GetProperty().SetColor(colors[0], colors[1], colors[2])

                actor.GetProperty().SetLineWidth(3)

                # assign actor to the renderer
                renderer.AddActor(actor)

    def saveResults(self, filename, imp):
        """
        Save result coordinates to specified file
        :param filename: file name
        :return: None
        """
        f = open(filename, 'wb')
        for i in range(self.joints.shape[0]):
            jt = imp.joints3DToImg(self.joints[i])
            for j in range(jt.shape[0]):
                for d in range(3):
                    f.write("{0:.3f} ".format(jt[j, d]))

            f.write("\n")

        f.close()

    def saveVideo(self, filename, sequence, imp, showJoints=True, showDepth=True, fullFrame=True, plotFrameNumbers=False,
                  visibility=None, highlight=None, height=320, width=240, joints2D_override=None, showGT=False,
                  annoscale=1):
        """
        Create a video with 2D annotations
        :param filename: name of file to save
        :param sequence: sequence data
        :param imp: importer of data
        :param showJoints: show joints
        :param showDepth: show depth map as background
        :param fullFrame: display full frame or only cropped frame
        :param plotFrameNumbers: plot frame numbers and indicate reference frames
        :param visibility: plot different markers for visible/non-visible joints, visible are marked as 1, non-visible 0
        :param highlight: highlight different joints, set as 1 to highlight, otherwise 0
        :return: None
        """

        if visibility is not None:
            assert visibility.shape == self.joints.shape, "{} == {}".format(visibility.shape, self.joints.shape)

        if highlight is not None:
            assert highlight.shape == self.joints.shape,  "{} == {}".format(highlight.shape, self.joints.shape)

        if joints2D_override is not None:
            assert joints2D_override.shape == self.joints.shape,  "{} == {}".format(joints2D_override.shape, self.joints.shape)

        # check aspect ratio
        if fullFrame:
            dpt = imp.loadDepthMap(sequence.data[0].fileName)
        else:
            dpt = sequence.data[0].dpt
        if dpt.shape[0] == dpt.shape[1]:
            width = height

        txt = 'Saving {}'.format(filename)
        pbar = pb.ProgressBar(maxval=self.joints.shape[0], widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()

        # Define the codec and create VideoWriter object
        fourcc = cv2.cv.CV_FOURCC(*'DIVX')
        video = cv2.VideoWriter('{}/video_{}.avi'.format(self.subfolder, filename), fourcc, self.fps, (height, width))
        if not video:
            raise EnvironmentError("Error in creating video writer")

        for i in xrange(self.joints.shape[0]):
            if joints2D_override is not None:
                jtI = joints2D_override[i]
            else:
                jt = self.joints[i]
                jtI = imp.joints3DToImg(jt)
            if fullFrame:
                if not os.path.isfile(sequence.data[i].fileName):
                    raise EnvironmentError("file missing")
                dm = imp.loadDepthMap(sequence.data[i].fileName)
                dm[dm == imp.getDepthMapNV()] = 0
                dpt = dm
                gtcrop = imp.joints3DToImg(self.gt[i])
            else:
                for joint in xrange(jtI.shape[0]):
                    jtI[joint, 0:2] = transformPoint2D(jtI[joint], sequence.data[i].T).squeeze()[0:2]
                dpt = sequence.data[i].dpt
                gtcrop = imp.joints3DToImg(self.gt[i])
                for joint in xrange(gtcrop.shape[0]):
                    gtcrop[joint, 0:2] = transformPoint2D(gtcrop[joint], sequence.data[i].T).squeeze()[0:2]
            img = self.plotResult(dpt, gtcrop, jtI, showGT=showGT, niceColors=True, showJoints=showJoints,
                                  showDepth=showDepth, visibility=None if visibility is None else visibility[i],
                                  highlight=None if highlight is None else highlight[i], annoscale=annoscale)
            img = img[:, :, [2, 1, 0]]  # change color channels
            img = cv2.resize(img, (height, width))
            if plotFrameNumbers:
                if sequence.data[i].subSeqName == 'ref':
                    cv2.putText(img, "Reference Frame {}".format(i), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
                # plot frame number
                cv2.putText(img, "{}".format(i), (height-50, width-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
            # write frame
            video.write(img)
            pbar.update(i)

        video.release()
        del video
        cv2.destroyAllWindows()
        pbar.finish()

    def saveVideo3D(self, filename, sequence, showPC=True, showGT=False, niceColors=True, plotFrameNumbers=False,
                    height=400, width=400):
        """
        Create a video with 3D annotations
        :param filename: name of file to save
        :param sequence: sequence data
        :return: None
        """

        txt = 'Saving {}'.format(filename)
        pbar = pb.ProgressBar(maxval=self.joints.shape[0], widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()

        # Define the codec and create VideoWriter object
        fourcc = cv2.cv.CV_FOURCC(*'DIVX')
        video = cv2.VideoWriter('{}/depth_{}.avi'.format(self.subfolder, filename), fourcc, self.fps, (height, width))
        if not video:
            raise EnvironmentError("Error in creating video writer")

        for i in range(self.joints.shape[0]):
            jt = self.joints[i]
            img = self.plotResult3D_OS(sequence.data[i].dpt, sequence.data[i].T, sequence.data[i].gt3Dorig, jt,
                                       showPC=showPC, showGT=showGT, niceColors=niceColors, width=width, height=height)
            img = numpy.flipud(img)
            img = img[:, :, [2, 1, 0]]  # change color channels for OpenCV
            img = cv2.resize(img, (height, width))
            if plotFrameNumbers:
                if sequence.data[i].subSeqName == 'ref':
                    cv2.putText(img, "Reference Frame {}".format(i), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
                # plot frame number
                cv2.putText(img, "{}".format(i), (height-50, width-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
            # write frame
            video.write(img)
            pbar.update(i)

        video.release()
        del video
        cv2.destroyAllWindows()
        pbar.finish()

    def saveVideoFrames(self, filename, images):
        """
        Create a video with synthesized images
        :param filename: name of file to save
        :param images: video data
        :return: None
        """

        txt = 'Saving {}'.format(filename)
        pbar = pb.ProgressBar(maxval=images.shape[0], widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()

        height = width = 128

        # Define the codec and create VideoWriter object
        fourcc = cv2.cv.CV_FOURCC(*'DIVX')
        video = cv2.VideoWriter('{}/synth_{}.avi'.format(self.subfolder, filename), fourcc, self.fps, (height, width))
        if not video:
            raise EnvironmentError("Error in creating video writer")

        for i in range(images.shape[0]):
            img = images[i]
            img = cv2.normalize(img, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8UC1)
            img = cv2.cvtColor(img, cv2.cv.CV_GRAY2BGR)
            img = cv2.resize(img, (height, width))
            # write frame
            video.write(img)
            pbar.update(i)

        video.release()
        del video
        cv2.destroyAllWindows()

        pbar.finish()

    def rgb_to_gray(self, rgb):
        """
        Convert rgb color to gray
        """
        assert len(rgb) == 3, "rgb should be 3, got {}".format(len(rgb))
        g = 0.21*rgb[0] + 0.72*rgb[1] + 0.07*rgb[2]
        return numpy.asarray([g, g, g])


class Blender2HandposeEvaluation(HandposeEvaluation):
    """
    Different evaluation metrics for handpose specific for Blender dataset
    """

    def __init__(self, gt, joints, dolegend=True, linewidth=1):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        super(Blender2HandposeEvaluation, self).__init__(gt, joints, dolegend, linewidth)
        import matplotlib.colors

        # setup specific stuff
        self.jointNames = ['CT', 'T1', 'T2', 'T3', 'CI', 'I1', 'I2', 'I3', 'I4', 'CM', 'M1', 'M2', 'M3', 'M4',
                           'CR', 'R1', 'R2', 'R3', 'R4', 'CP', 'P1', 'P2', 'P3', 'P4']
        self.jointColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.2]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.2]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.2]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.2]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1.0]]]))[0, 0]]
        self.jointConnections = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
                                 [12, 13], [14, 15], [15, 16], [16, 17], [17, 18], [19, 20], [20, 21], [21, 22],
                                 [22, 23]]
        self.jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0]]

        self.plotMaxJointDist = 80
        self.VTKviewport = [0, 0, 180, 0, 50]
        self.fps = 10.0

    def getPCL(self, dpt, T):
        """
        Get pointcloud from frame
        :param dpt: depth image
        :param T: 2D transformation of crop
        """

        return Blender2Importer.depthToPCL(dpt, T)

