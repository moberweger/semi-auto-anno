"""Functions for interactive object detection.

InteractiveDetector provides interface for interactive object
detection.

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

import getpass
import time
import numpy
import os
import progressbar as pb
import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from util.handdetector import HandDetector


__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2016, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class InteractiveDetector(QMainWindow):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    @see http://matplotlib.org/examples/event_handling/data_browser.html
    @see http://matplotlib.org/examples/event_handling/poly_editor.html
    """

    def __init__(self, Seq, hpe, importer, filename_detection, filename_log, subset_idxs, start_idx, parent=None):
        super(InteractiveDetector, self).__init__(parent)
        self.setWindowTitle('Detection Tool')
        self.dpi = 100
        self.panned = False
        self.lastind = 0
        self._ind = None
        self.epsilon = 15
        self.press = (0, 0, 0, 0)
        self.hpe = hpe
        self.importer = importer
        self.path_prefix = ''
        if filename_log is not None:
            if os.path.exists(filename_log):
                self.file_log = open(filename_log, 'a')
            else:
                self.file_log = open(filename_log, 'w')
        else:
            self.file_log = None
        self.start_time = time.time()

        self.subset_idxs = subset_idxs

        if len(self.subset_idxs) > 0:
            self.curFrame = self.subset_idxs[0]
        else:
            self.curFrame = 0
        self._seq = Seq
        self.plots_xy = []
        self.plots_xz = []
        self.plots_yz = []
        self.plots_color = []

        self.getCurrentStatus(filename_detection)

        if start_idx != 0:
            if len(self.subset_idxs) > 0:
                if start_idx in self.subset_idxs:
                    self.curFrame = start_idx
                else:
                    raise UserWarning("Unknown start_idx, not in subset!")
            else:
                self.curFrame = start_idx

        if ~numpy.allclose(self._seq.data[self.curFrame].com, 0.):
            self.curData = self.importer.joint3DToImg(self._seq.data[self.curFrame].com).reshape((1, 3))
        else:
            self.curData = numpy.zeros((1, 3))

        self.filename_det = filename_detection

        self.initGUI()
        self.connectEvents()

        self.showCurrent()

    def initGUI(self):
        self.main_frame = QWidget(self)

        # XY plot and toolbar
        self.fig_xy = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas_xy = FigureCanvas(self.fig_xy)
        self.canvas_xy.setParent(self.main_frame)
        self.ax_xy = self.fig_xy.add_subplot(111)
        self.ax_xy.set_title('Click and drag a point to move it')
        box = self.ax_xy.get_position()
        self.ax_xy.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        self.mpl_toolbar_xy = NavigationToolbar(self.canvas_xy, self.main_frame)

        def pan_callback():
            self.panned = True
        self.fig_xy.canvas.toolbar.actions()[4].triggered.connect(pan_callback)

        # XZ, YZ plot and toolbar
        self.fig_xz_yz = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas_xz_yz = FigureCanvas(self.fig_xz_yz)
        self.canvas_xz_yz.setParent(self.main_frame)
        gs = matplotlib.gridspec.GridSpec(2, 1)
        self.ax_xz = self.fig_xz_yz.add_subplot(gs[0])
        self.ax_xz.axes.set_frame_on(False)
        self.ax_yz = self.fig_xz_yz.add_subplot(gs[1])
        self.ax_yz.axes.set_frame_on(False)
        self.mpl_toolbar_xz_yz = NavigationToolbar(self.canvas_xz_yz, self.main_frame)

        # optional color plot and toolbar
        if hasattr(self._seq.data[self.curFrame], 'colorFileName'):
            self.fig_color = Figure((5.0, 4.0), dpi=self.dpi)
            self.canvas_color = FigureCanvas(self.fig_color)
            self.canvas_color.setParent(self.main_frame)
            self.ax_color = self.fig_color.add_subplot(111)
            self.ax_color.axes.set_frame_on(False)
            self.mpl_toolbar_color = NavigationToolbar(self.canvas_color, self.main_frame)
        else:
            self.ax_color = None

        # Layout with posebit controls
        hbox_pb_controls = QHBoxLayout()

        self.prevButton = QPushButton('Previous', self.main_frame)
        self.connect(self.prevButton, SIGNAL('clicked()'), self.prevButton_callback)
        hbox_pb_controls.addWidget(self.prevButton)

        self.pointcloudButton = QPushButton('3D View', self.main_frame)
        self.connect(self.pointcloudButton, SIGNAL('clicked()'), self.pointcloudButton_callback)
        hbox_pb_controls.addWidget(self.pointcloudButton)

        self.nextButton = QPushButton('Next', self.main_frame)
        self.connect(self.nextButton, SIGNAL('clicked()'), self.nextButton_callback)
        hbox_pb_controls.addWidget(self.nextButton)

        hbox_plots = QHBoxLayout()

        vbox_xy = QVBoxLayout()
        vbox_xy.addWidget(self.canvas_xy)
        vbox_xy.addWidget(self.mpl_toolbar_xy)
        hbox_plots.addLayout(vbox_xy)

        vbox_xz_yz = QVBoxLayout()
        vbox_xz_yz.addWidget(self.canvas_xz_yz)
        vbox_xz_yz.addWidget(self.mpl_toolbar_xz_yz)
        hbox_plots.addLayout(vbox_xz_yz)

        if self.ax_color is not None:
            vbox_color = QVBoxLayout()
            vbox_color.addWidget(self.canvas_color)
            vbox_color.addWidget(self.mpl_toolbar_color)
            hbox_plots.addLayout(vbox_color)

        vbox_all = QVBoxLayout()
        vbox_all.addLayout(hbox_plots)
        sep = QFrame(self.main_frame)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        vbox_all.addWidget(sep)
        vbox_all.addLayout(hbox_pb_controls)

        self.main_frame.setLayout(vbox_all)
        self.setCentralWidget(self.main_frame)

    def showCurrent(self):
        dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
        if ~numpy.allclose(self.curData, 0.):
            dm[dm < self.curData[0, 2]-self._seq.config['cube'][2]/2.] = self.curData[0, 2] + self._seq.config['cube'][2]/2.
            dm[dm > self.curData[0, 2]+self._seq.config['cube'][2]/2.] = self.curData[0, 2] + self._seq.config['cube'][2]/2.

        # XY plot
        if self.panned is False:
            xstart = numpy.where(~numpy.isclose(dm, self.curData[0, 2] + self._seq.config['cube'][2] / 2.))[1].min()
            xend = numpy.where(~numpy.isclose(dm, self.curData[0, 2] + self._seq.config['cube'][2] / 2.))[1].max()
            ystart = numpy.where(~numpy.isclose(dm, self.curData[0, 2] + self._seq.config['cube'][2] / 2.))[0].min()
            yend = numpy.where(~numpy.isclose(dm, self.curData[0, 2] + self._seq.config['cube'][2] / 2.))[0].max()
        else:
            xstart, xend = self.ax_xy.get_xlim()
            yend, ystart = self.ax_xy.get_ylim()  # axis is flipped after imshow
        self.ax_xy.cla()
        self.text = self.ax_xy.text(0.05, 0.95, 'joint: none', transform=self.ax_xy.transAxes, va='top')
        self.frame = self.ax_xy.text(0.55, 0.95, 'frame: none', transform=self.ax_xy.transAxes, va='top')
        self.selected, = self.ax_xy.plot([0], [0], 'o', ms=12, alpha=0.4, color='yellow', visible=False)
        self.frame.set_text('frame: {}/{}'.format(self.curFrame, len(self._seq.data) - 1))
        self.ax_xy.imshow(dm, cmap='CMRmap', interpolation='none')
        self.ax_xy.set_xlabel('x')
        self.ax_xy.set_ylabel('y')
        self.plots_xy = []
        for i in range(self.curData.shape[0]):
            d, = self.ax_xy.plot(self.curData[i, 0], self.curData[i, 1], c='r', marker='o', markersize=8)
            self.plots_xy.append(d)

        self.ax_xy.set_xlim([xstart, xend])
        self.ax_xy.set_ylim([yend, ystart])

        def format_coord(x, y):
            numrows, numcols = dm.shape
            col = int(numpy.rint(x))
            row = int(numpy.rint(y))
            if 0 <= col < numcols and 0 <= row < numrows:
                z = dm[row, col]
                return 'x=%d,y=%d,z=%d' % (x, y, z)
            else:
                return 'x=%d,y=%d' % (x, y)

        self.ax_xy.format_coord = format_coord

        # XZ plot
        self.ax_xz.cla()
        xz = numpy.min(dm, axis=0)
        self.ax_xz.plot(numpy.arange(dm.shape[1]), xz)
        self.ax_xz.set_xlabel('x')
        self.ax_xz.set_ylabel('z')
        pts = numpy.column_stack((numpy.arange(dm.shape[1] + 2),
                                  numpy.concatenate([numpy.asarray([0]), xz, numpy.asarray([0])])))
        polygon = Polygon(pts, True)
        p = PatchCollection([polygon], cmap='jet', alpha=0.4)
        p.set_array(numpy.array([50, 50, 50, 50]))
        self.ax_xz.add_collection(p)

        self.plots_xz = []
        for i in range(self.curData.shape[0]):
            d, = self.ax_xz.plot(self.curData[i, 0], self.curData[i, 2], c='r', marker='o', markersize=8)
            self.plots_xz.append(d)

        # YZ plot
        self.ax_yz.cla()
        yz = numpy.min(dm, axis=1)
        self.ax_yz.plot(numpy.arange(dm.shape[0]), yz)
        self.ax_yz.set_xlabel('y')
        self.ax_yz.set_ylabel('z')
        pts = numpy.column_stack((numpy.arange(dm.shape[0] + 2),
                                  numpy.concatenate([numpy.asarray([0]), yz, numpy.asarray([0])])))
        polygon = Polygon(pts, True)
        p = PatchCollection([polygon], cmap='jet', alpha=0.4)
        p.set_array(numpy.array([50, 50, 50, 50]))
        self.ax_yz.add_collection(p)

        self.plots_yz = []
        for i in range(self.curData.shape[0]):
            d, = self.ax_yz.plot(self.curData[i, 1], self.curData[i, 2], c='r', marker='o', markersize=8)
            self.plots_yz.append(d)

        # color image
        if self.ax_color:
            color = self.importer.loadColorImage(self.path_prefix+self._seq.data[self.curFrame].colorFileName)

            # XY plot
            self.ax_color.cla()
            if len(color.shape) == 3:
                self.ax_color.imshow(color.astype('uint8'), interpolation='none')
            else:
                self.ax_color.imshow(color, interpolation='none', cmap='gray')
            self.ax_color.set_xlabel('u')
            self.ax_color.set_ylabel('v')

            self.plots_color = []
            joint2D = self.importer.jointsDpt2DToCol2D(self.curData)
            for i in range(joint2D.shape[0]):
                d, = self.ax_color.plot(joint2D[i, 0], joint2D[i, 1], c='r', marker='o', markersize=8)
                self.plots_color.append(d)

        self.update()

    def showContext(self):
        dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
        dm[dm < self.curData[0, 2] - self._seq.config['cube'][2] / 2.] = \
            self.curData[0, 2] + self._seq.config['cube'][2] / 2.
        dm[dm > self.curData[0, 2] + self._seq.config['cube'][2] / 2.] = \
            self.curData[0, 2] + self._seq.config['cube'][2] / 2.

        self.ax_xz.cla()
        xz = dm[int(numpy.rint(self.curData[self.lastind, 1]))]
        self.ax_xz.plot(numpy.arange(dm.shape[1]), xz)
        self.plots_xz = []
        d, = self.ax_xz.plot(self.curData[self.lastind, 0], self.curData[self.lastind, 2], c='r',
                             marker='o', markersize=8)
        self.plots_xz.append(d)
        self.ax_xz.set_xlim([self.curData[self.lastind, 0] - self._seq.data[self.curFrame].dpt.shape[0] // 4,
                             self.curData[self.lastind, 0] + self._seq.data[self.curFrame].dpt.shape[0] // 4])
        self.ax_xz.set_ylim([self.curData[self.lastind, 2] - self._seq.config['cube'][2] // 4,
                             self.curData[self.lastind, 2] + self._seq.config['cube'][2] // 4])
        self.ax_xz.set_xlabel('x')
        self.ax_xz.set_ylabel('z')

        self.ax_yz.cla()
        yz = dm[:, int(numpy.rint(self.curData[self.lastind, 0]))]
        self.ax_yz.plot(numpy.arange(dm.shape[0]), yz)
        self.plots_yz = []
        d, = self.ax_yz.plot(self.curData[self.lastind, 1], self.curData[self.lastind, 2], c='r',
                             marker='o', markersize=8)
        self.plots_yz.append(d)
        self.ax_yz.set_xlim([self.curData[self.lastind, 1] - self._seq.data[self.curFrame].dpt.shape[1] // 4,
                             self.curData[self.lastind, 1] + self._seq.data[self.curFrame].dpt.shape[1] // 4])
        self.ax_yz.set_ylim([self.curData[self.lastind, 2] - self._seq.config['cube'][2] // 4,
                             self.curData[self.lastind, 2] + self._seq.config['cube'][2] // 4])
        self.ax_yz.set_xlabel('y')
        self.ax_yz.set_ylabel('z')

        self.update()

    def getCurrentStatus(self, filename_detections):
        pbar = pb.ProgressBar(maxval=len(self._seq.data), widgets=['Loading last status', pb.Percentage(), pb.Bar()])
        pbar.start()
        cache_str = ''
        with open(filename_detections, "r") as inputfile:
            cache_str = inputfile.readlines()

        for i in xrange(len(self._seq.data)):
            pbar.update(i)
            if len(self.subset_idxs) > 0:
                if i not in self.subset_idxs:
                    break

            hd = HandDetector(numpy.zeros((1, 1)), 0., 0.)  # dummy object
            com = numpy.asarray(hd.detectFromCache(filename_detections, self._seq.data[i].fileName, cache_str))
            if numpy.allclose(com[2], 0.):
                self.curFrame = i
                break
            else:
                self._seq.data[i] = self._seq.data[i]._replace(com=self.importer.jointImgTo3D(com.reshape((3,))))

        # redo last pose, it might be set to default and saved
        if self.curFrame > 0:
            if len(self.subset_idxs) > 0:
                if self.subset_idxs.index(self.curFrame) - 1 >= 0:
                    self.curFrame = self.subset_idxs[self.subset_idxs.index(self.curFrame) - 1]
            else:
                self.curFrame -= 1

    def saveCurrent(self):

        self._seq.data[self.curFrame] = self._seq.data[self.curFrame]._replace(com=self.importer.jointImgTo3D(self.curData.reshape((3,))))

        if self.filename_det is not None:
            self.importer.saveSequenceDetections(self._seq, self.filename_det)

        # save log file
        if self.file_log is not None:
            self.file_log.write('{}, {}, {}@{}, {}\n'.format(self._seq.data[self.curFrame].fileName,
                                                             time.strftime("%d.%m.%Y %H:%M:%S +0000", time.gmtime()),
                                                             getpass.getuser(),
                                                             os.uname()[1],
                                                             time.time() - self.start_time))
            self.file_log.flush()

    def connectEvents(self):
        self.canvas_xy.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas_xy.mpl_connect('button_release_event', self.button_release_callback)
        self.canvas_xy.mpl_connect('motion_notify_event', self.motion_notify_callback)

    def save_plot(self):
        file_choices = "PNG (*.png)|*.png"

        path = unicode(QFileDialog.getSaveFileName(self, 'Save file', '', file_choices))
        if path:
            self.canvas_xy.print_figure(path, dpi=self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)

    def pointcloudButton_callback(self):
        self.pointcloud()

    def nextButton_callback(self):
        self.next()

    def prevButton_callback(self):
        self.prev()

    def keyPressEvent(self, event):
        if self.lastind is None:
            return

        if event.text() == 'q':
            QApplication.instance().quit()
        elif event.text() == 'p':
            # previous
            self.prev()
            print 'prev'
        elif event.text() == 'n':
            # next
            self.next()
            print 'next'
        elif event.text() == '3':
            # pointcloud
            self.pointcloud()
            print '3D'
        elif event.text() == 'r':
            # reload
            self.panned = False
            self.showCurrent()
            print 'reload'
        elif event.text() == 's':
            # save
            if self.filename_det is not None:
                self.importer.saveSequenceDetections(self._seq, self.filename_det)
            print 'saved'
        elif event.text() == 'a':
            # snap to z
            if self.lastind is not None:
                self.snap_z(self.lastind)
                print 'align'
                self.showContext()
        elif event.text() == 'z':
            # snap to z
            self.snap_z()
            print 'snap'
            self.showContext()
        elif event.key() == Qt.Key_Plus:
            self.curData[self.lastind, 2] += 5.
            print "z+=5"
            self.showContext()
        elif event.key() == Qt.Key_Minus:
            self.curData[self.lastind, 2] -= 5.
            print "z-=5"
            self.showContext()
        elif event.key() == Qt.Key_Up:
            self.curData[self.lastind, 1] -= 1.
            print "x-=1"
            self.showCurrent()
        elif event.key() == Qt.Key_Down:
            self.curData[self.lastind, 1] += 1.
            print "x+=1"
            self.showCurrent()
        elif event.key() == Qt.Key_Right:
            self.curData[self.lastind, 0] += 1.
            print "y+=1"
            self.showCurrent()
        elif event.key() == Qt.Key_Left:
            self.curData[self.lastind, 0] -= 1.
            print "y-=1"
            self.showCurrent()
        else:
            return

        self.update()

    def snap_z(self, idx=None):
        dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
        if idx is None:
            for i in range(self.curData.shape[0]):
                self.curData[i, 2] = dm[int(numpy.rint(self.curData[i, 1])), int(numpy.rint(self.curData[i, 0]))]
        else:
            self.curData[idx, 2] = dm[int(numpy.rint(self.curData[idx, 1])), int(numpy.rint(self.curData[idx, 0]))]

    def pointcloud(self):
        dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
        cur3D = self.importer.jointImgTo3D(self.curData[0]).reshape((1, 3))
        hd = HandDetector(dm, self.importer.fx, self.importer.fy, importer=self.importer)
        dpt, M, com = hd.cropArea3D(self.curData[0].reshape((3,)), size=self._seq.config['cube'], docom=False)
        self.hpe.plotResult3D(dpt, M, cur3D, cur3D, showGT=False, niceColors=False)

    def next(self):
        self.panned = False
        self.saveCurrent()
        if len(self.subset_idxs) > 0:
            if len(self.subset_idxs) > self.subset_idxs.index(self.curFrame) + 1:
                self.curFrame = self.subset_idxs[self.subset_idxs.index(self.curFrame) + 1]
            else:
                print "Done"
                QApplication.instance().quit()
        else:
            if self.curFrame < len(self._seq.data) - 1:
                self.curFrame += 1
            else:
                print "Done"
                QApplication.instance().quit()

        # reset timer
        self.start_time = time.time()

        # if there is no data, use the previous data to initialize
        com = self._seq.data[self.curFrame].com
        if numpy.allclose(com, 0.):
            prevFrame = self.curFrame
            if len(self.subset_idxs) > 0:
                if self.subset_idxs.index(self.curFrame) - 1 >= 0:
                    prevFrame = self.subset_idxs[self.subset_idxs.index(self.curFrame) - 1]
            else:
                if self.curFrame > 0:
                    prevFrame = self.curFrame - 1
            prev_com = self._seq.data[prevFrame].com
            if numpy.allclose(prev_com, 0.):
                self.curData = numpy.zeros((1, 3))
            else:
                self.curData = self.importer.joint3DToImg(prev_com).reshape((1, 3))
        else:
            self.curData = self.importer.joint3DToImg(com).reshape((1, 3))
        self.showCurrent()

    def prev(self):
        self.panned = False
        self.saveCurrent()
        if len(self.subset_idxs) > 0:
            if self.subset_idxs.index(self.curFrame) - 1 >= 0:
                self.curFrame = self.subset_idxs[self.subset_idxs.index(self.curFrame) - 1]
            else:
                print "First already selected!"
        else:
            if self.curFrame > 0:
                self.curFrame -= 1
            else:
                print "First already selected!"

        # reset timer
        self.start_time = time.time()

        prev_com = self._seq.data[self.curFrame].com
        if numpy.allclose(prev_com, 0.):
            self.curData = numpy.zeros((1, 3))
        else:
            self.curData = self.importer.joint3DToImg(prev_com).reshape((1, 3))
        self.showCurrent()

    def update(self):
        if self.lastind is None:
            return

        self.selected.set_visible(True)
        if self._ind is not None:
            self.selected.set_data(self.curData[self._ind, 0], self.curData[self._ind, 1])

        self.text.set_text('joint: %d' % self.lastind)
        self.ax_xy.get_figure().canvas.draw()
        self.ax_xz.get_figure().canvas.draw()
        self.ax_yz.get_figure().canvas.draw()
        if self.ax_color:
            if self._ind is not None:
                j2D = self.importer.jointsDpt2DToCol2D(self.curData)
                self.plots_color[self._ind].set_data(j2D[self._ind, 0], j2D[self._ind, 1])

            self.ax_color.get_figure().canvas.draw()

    def get_ind_under_point(self, event):
        """
        get the index of the vertex under point if within epsilon tolerance
        :param event: qt event
        :return: index of selected point
        """

        # display coords
        distances = numpy.hypot(event.xdata - self.curData[:, 0],
                                event.ydata - self.curData[:, 1])
        indmin = distances.argmin()

        if distances[indmin] >= self.epsilon:
            ind = None
        else:
            ind = indmin
        self.lastind = ind
        return ind

    def button_press_callback(self, event):
        """
        whenever a mouse button is pressed
        :param event: qt event
        :return:
        """
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)
        print "got joint id", self._ind

        # if there is only one joint
        if self._ind is None and self.curData.shape[0] == 1:
            self._ind = 0
            self.lastind = 0
            self.plots_xy[self._ind].set_data([event.xdata, event.ydata])

        if self._ind is not None:
            x0, y0 = self.plots_xy[self._ind].get_data()
            self.press = x0, y0, event.xdata, event.ydata
        self.update()

    def button_release_callback(self, event):
        """
        whenever a mouse button is released
        :param event: qt event
        :return:
        """
        if event.button != 1:
            return
        if self._ind is None:
            return
        x, y = self.plots_xy[self._ind].get_data()
        self.curData[self._ind, 0] = x
        self.curData[self._ind, 1] = y

        # set to closest valid depth if at invalid depth
        dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
        if numpy.isclose(dm[int(y), int(x)], self.importer.getDepthMapNV()):
            validmsk = ~numpy.isclose(dm, self.importer.getDepthMapNV())
            pts = numpy.asarray(numpy.where(validmsk)).transpose()
            deltas = pts - numpy.asarray([int(y), int(x)])
            dist = numpy.einsum('ij,ij->i', deltas, deltas)
            pos = numpy.argmin(dist)
            self.curData[self._ind, 2] = dm[pts[pos][0], pts[pos][1]] + 10  # add offset to depth
        else:
            self.curData[self._ind, 2] = dm[int(y), int(x)] + 10  # add offset to depth
        self._ind = None
        self.showCurrent()
        self.showContext()

    def motion_notify_callback(self, event):
        """
        on mouse movement
        :param event: qt event
        :return:
        """
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.selected.set_data(x0 + dx, y0 + dy)
        self.plots_xy[self._ind].set_data(x0 + dx, y0 + dy)
        self.ax_xy.get_figure().canvas.draw()
