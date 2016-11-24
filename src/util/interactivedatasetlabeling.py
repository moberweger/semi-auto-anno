"""Functions for interactive pose annotation.

InteractiveDatasetLabeling provides interface for interactive pose
annotation.

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

import gc
import getpass
import time
import numpy
import os
import progressbar as progb
import matplotlib
matplotlib.rcParams['legend.handlelength'] = 0
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from PyQt4.QtCore import *
from PyQt4.QtGui import *


__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2016, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class InteractiveDatasetLabeling(QMainWindow):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    @see http://matplotlib.org/examples/event_handling/data_browser.html
    @see http://matplotlib.org/examples/event_handling/poly_editor.html
    """

    def __init__(self, Seq, hpe, importer, hc, filename_joints, filename_pb, filename_vis, filename_log, subset_idxs,
                 start_idx, replace_file=None, replace_off=0, parent=None):
        super(InteractiveDatasetLabeling, self).__init__(parent)
        self.setWindowTitle('Pose Annotation Tool')
        self.dpi = 100
        self.panned = False
        self.lastind = 0
        self._ind = None
        self.epsilon = 5
        self.press = (0, 0, 0, 0)
        self.hpe = hpe
        self.importer = importer
        self.hc = hc
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
        self.replace_file = replace_file
        self.replace_off = replace_off

        self.curCorrected = []
        if len(self.subset_idxs) > 0:
            self.curFrame = self.subset_idxs[0]
        else:
            self.curFrame = 0
        self._seq = Seq
        self.plots_xy = []
        self.plots_xz = []
        self.plots_yz = []
        self.plots_color = []

        if filename_joints is not None and filename_pb is not None and filename_vis is not None:
            self.getCurrentStatus(filename_joints, filename_pb, filename_vis)

        if start_idx != 0:
            if len(self.subset_idxs) > 0:
                if start_idx in self.subset_idxs:
                    self.curFrame = start_idx
                else:
                    raise UserWarning("Unknown start_idx, not in subset!")
            else:
                self.curFrame = start_idx

        if self.replace_file is not None:
            data = numpy.load(self.replace_file)
            if isinstance(data, numpy.lib.npyio.NpzFile):
                dat = data['arr_1'][self.curFrame-self.replace_off].reshape(self._seq.data[self.curFrame].gtorig.shape)*self._seq.config['cube'][2]/2.+self._seq.data[self.curFrame].com
            else:
                dat = data[self.curFrame-self.replace_off].reshape(self._seq.data[self.curFrame].gtorig.shape)*self._seq.config['cube'][2]/2.+self._seq.data[self.curFrame].com
            self.curData = self.importer.joints3DToImg(dat)
        else:
            if ~numpy.allclose(self._seq.data[self.curFrame].gtorig, 0.):
                self.curData = self._seq.data[self.curFrame].gtorig.copy()
            else:
                self.curData = self.importer.default_gtorig.copy() + self.importer.joint3DToImg(self._seq.data[self.curFrame].com)

        if hasattr(self._seq.data[self.curFrame], 'extraData') and self._seq.data[self.curFrame].extraData:
            if 'pb' in self._seq.data[self.curFrame].extraData.keys():
                self.curPb = self._seq.data[self.curFrame].extraData['pb']['pb']
                self.curPbP = self._seq.data[self.curFrame].extraData['pb']['pbp']
            else:
                self.curPb = []
                self.curPbP = []
                for p in range(len(self.hc.posebits)):
                    if abs(self.curData[self.hc.posebits[p][0], 2] - self.curData[self.hc.posebits[p][1], 2]) > self.hc.pb_thresh:
                        if self.curData[self.hc.posebits[p][0], 2] < self.curData[self.hc.posebits[p][1], 2]:
                            self.curPb.append((self.hc.posebits[p][0], self.hc.posebits[p][1]))
                            self.curPbP.append(self.hc.posebits[p])
                        else:
                            self.curPb.append((self.hc.posebits[p][1], self.hc.posebits[p][0]))
                            self.curPbP.append(self.hc.posebits[p])

            if 'vis' in self._seq.data[self.curFrame].extraData.keys():
                self.curVis = self._seq.data[self.curFrame].extraData['vis']
                # default all visible
                if len(self.curVis) == 0:
                    self.curVis = range(0, self.curData.shape[0])
            else:
                dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
                self.curVis = self.importer.visibilityTest(dm, self.curData, 10.)
        else:
            self.curPb = []
            self.curPbP = []
            for p in range(len(self.hc.posebits)):
                if abs(self.curData[self.hc.posebits[p][0], 2] - self.curData[self.hc.posebits[p][1], 2]) > self.hc.pb_thresh:
                    if self.curData[self.hc.posebits[p][0], 2] < self.curData[self.hc.posebits[p][1], 2]:
                        self.curPb.append((self.hc.posebits[p][0], self.hc.posebits[p][1]))
                        self.curPbP.append(self.hc.posebits[p])
                    else:
                        self.curPb.append((self.hc.posebits[p][1], self.hc.posebits[p][0]))
                        self.curPbP.append(self.hc.posebits[p])
            dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
            self.curVis = self.importer.visibilityTest(dm, self.curData, 10.)

        self.filename_joints = filename_joints
        self.filename_pb = filename_pb
        self.filename_vis = filename_vis

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

        # Other GUI controls
        self.cb_pb = []
        self.rb_pb = []

        # Layout with posebit controls
        hbox_pb_controls = QHBoxLayout()

        self.prevButton = QPushButton('Previous', self.main_frame)
        self.connect(self.prevButton, SIGNAL('clicked()'), self.prevButton_callback)
        hbox_pb_controls.addWidget(self.prevButton)

        for w in range(len(self.hc.posebits)):
            self.cb_pb.append(QCheckBox(str(self.hc.posebits[w]), self.main_frame))
            self.cb_pb[-1].setChecked(False)
            self.connect(self.cb_pb[-1], SIGNAL('stateChanged(int)'), self.cb_pb_changed_callback)

            vbox = QVBoxLayout()
            vbox.addWidget(self.cb_pb[-1])

            # Number group
            r_group = QButtonGroup()
            r0 = QRadioButton("{} < {}".format(self.hc.posebits[w][0], self.hc.posebits[w][1]), self.main_frame)
            self.connect(r0, SIGNAL('toggled(bool)'), self.rb_pb_changed_callback)
            r0.setChecked(True)
            r_group.addButton(r0)
            r1 = QRadioButton("{} < {}".format(self.hc.posebits[w][1], self.hc.posebits[w][0]), self.main_frame)
            self.connect(r1, SIGNAL('toggled(bool)'), self.rb_pb_changed_callback)
            r_group.addButton(r1)
            vbox.addWidget(r0)
            vbox.addWidget(r1)
            self.rb_pb.append(r_group)

            hbox_pb_controls.addLayout(vbox)
            hbox_pb_controls.setAlignment(self.cb_pb[-1], Qt.AlignHCenter)

        vbox_right = QVBoxLayout()
        self.solveButton = QPushButton('Solve', self.main_frame)
        self.connect(self.solveButton, SIGNAL('clicked()'), self.solveButton_callback)
        self.nextButton = QPushButton('Next', self.main_frame)
        self.connect(self.nextButton, SIGNAL('clicked()'), self.nextButton_callback)
        vbox_right.addWidget(self.solveButton)
        vbox_right.addWidget(self.nextButton)
        hbox_pb_controls.addLayout(vbox_right)

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
        dm[dm < self._seq.data[self.curFrame].com[2] - self._seq.config['cube'][2] / 2.] = \
            self._seq.data[self.curFrame].com[2] + self._seq.config['cube'][2] / 2.
        dm[dm > self._seq.data[self.curFrame].com[2] + self._seq.config['cube'][2] / 2.] = \
            self._seq.data[self.curFrame].com[2] + self._seq.config['cube'][2] / 2.

        # XY plot
        if self.panned is False:
            arr = numpy.where(~numpy.isclose(dm, self._seq.data[self.curFrame].com[2] + self._seq.config['cube'][2] / 2.))[1]
            if arr.size > 0:
                xstart, xend = (arr.min(), arr.max())
            else:
                xstart, xend = (0, dm.shape[0])

            arr = numpy.where(~numpy.isclose(dm, self._seq.data[self.curFrame].com[2] + self._seq.config['cube'][2] / 2.))[0]
            if arr.size > 0:
                ystart, yend = (arr.min(), arr.max())
            else:
                ystart, yend = (0, dm.shape[1])
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
            d, = self.ax_xy.plot(self.curData[i, 0], self.curData[i, 1], c=self.hpe.jointColors[i],
                                 marker='o' if i in self.curVis else 'v', markersize=8)
            self.plots_xy.append(d)
            self.ax_xy.annotate(i, (self.curData[i, 0], self.curData[i, 1]))

        for i in range(len(self.hpe.jointConnections)):
            self.ax_xy.plot(numpy.hstack(
                (self.curData[self.hpe.jointConnections[i][0], 0], self.curData[self.hpe.jointConnections[i][1], 0])),
                numpy.hstack((self.curData[self.hpe.jointConnections[i][0], 1],
                              self.curData[self.hpe.jointConnections[i][1], 1])),
                c=self.hpe.jointConnectionColors[i], linewidth=2.0)

        self.ax_xy.set_xlim([xstart, xend])
        self.ax_xy.set_ylim([yend, ystart])
        self.ax_xy.legend(
            [plt.Line2D((0, 1), (0, 0), color='r', marker='o'), plt.Line2D((0, 1), (0, 0), color='r', marker='v')],
            ['Visible', 'Not visible'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, numpoints=1)

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
            d, = self.ax_xz.plot(self.curData[i, 0], self.curData[i, 2], c=self.hpe.jointColors[i],
                                 marker='o' if i in self.curVis else 'v', markersize=8)
            self.plots_xz.append(d)

        for i in range(len(self.hpe.jointConnections)):
            self.ax_xz.plot(numpy.hstack(
                (self.curData[self.hpe.jointConnections[i][0], 0], self.curData[self.hpe.jointConnections[i][1], 0])),
                numpy.hstack((self.curData[self.hpe.jointConnections[i][0], 2],
                              self.curData[self.hpe.jointConnections[i][1], 2])),
                c=self.hpe.jointConnectionColors[i], linewidth=2.0)

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
            d, = self.ax_yz.plot(self.curData[i, 1], self.curData[i, 2], c=self.hpe.jointColors[i],
                                 marker='o' if i in self.curVis else 'v', markersize=8)
            self.plots_yz.append(d)

        for i in range(len(self.hpe.jointConnections)):
            self.ax_yz.plot(numpy.hstack(
                (self.curData[self.hpe.jointConnections[i][0], 1], self.curData[self.hpe.jointConnections[i][1], 1])),
                numpy.hstack((self.curData[self.hpe.jointConnections[i][0], 2],
                              self.curData[self.hpe.jointConnections[i][1], 2])),
                c=self.hpe.jointConnectionColors[i], linewidth=2.0)

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
                d, = self.ax_color.plot(joint2D[i, 0], joint2D[i, 1], c=self.hpe.jointColors[i],
                                        marker='o' if i in self.curVis else 'v', markersize=8)
                self.plots_color.append(d)
                self.ax_color.annotate(i, (joint2D[i, 0], joint2D[i, 1]))

            for i in range(len(self.hpe.jointConnections)):
                self.ax_color.plot(numpy.hstack(
                    (joint2D[self.hpe.jointConnections[i][0], 0], joint2D[self.hpe.jointConnections[i][1], 0])),
                    numpy.hstack((joint2D[self.hpe.jointConnections[i][0], 1],
                                  joint2D[self.hpe.jointConnections[i][1], 1])),
                    c=self.hpe.jointConnectionColors[i], linewidth=2.0)

            xstart = min(max(0, joint2D[0, 0] - self._seq.data[self.curFrame].color.shape[0] // 2), color.shape[0])
            self.ax_color.set_xlim([xstart, xstart + self._seq.data[self.curFrame].color.shape[1]])
            ystart = min(max(0, joint2D[0, 1] - self._seq.data[self.curFrame].color.shape[1] // 2), color.shape[1])
            self.ax_color.set_ylim([ystart + self._seq.data[self.curFrame].color.shape[0], ystart])

        # update posebit controls
        for idx, cb in enumerate(self.cb_pb):
            cb.blockSignals(True)
            cb.setChecked(False)
            for i, k in enumerate(self.curPbP):
                if str(k) == cb.text():
                    cb.setChecked(True)
                    if (self.curPb[i][0] == k[0]) and (self.curPb[i][1] == k[1]):
                        self.rb_pb[idx].buttons()[0].setChecked(True)
                    else:
                        self.rb_pb[idx].buttons()[1].setChecked(True)

            cb.blockSignals(False)

        self.update()

    def showContext(self):
        dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
        dm[dm < self._seq.data[self.curFrame].com[2] - self._seq.config['cube'][2] / 2.] = \
            self._seq.data[self.curFrame].com[2] + self._seq.config['cube'][2] / 2.
        dm[dm > self._seq.data[self.curFrame].com[2] + self._seq.config['cube'][2] / 2.] = \
            self._seq.data[self.curFrame].com[2] + self._seq.config['cube'][2] / 2.

        self.ax_xz.cla()
        xz = dm[int(numpy.rint(self.curData[self.lastind, 1]))]
        self.ax_xz.plot(numpy.arange(dm.shape[1]), xz)
        self.plots_xz = []
        d, = self.ax_xz.plot(self.curData[self.lastind, 0], self.curData[self.lastind, 2],
                             c=self.hpe.jointColors[self.lastind],
                             marker='o' if self.lastind in self.curVis else 'v', markersize=8)
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
        d, = self.ax_yz.plot(self.curData[self.lastind, 1], self.curData[self.lastind, 2],
                             c=self.hpe.jointColors[self.lastind],
                             marker='o' if self.lastind in self.curVis else 'v', markersize=8)
        self.plots_yz.append(d)
        self.ax_yz.set_xlim([self.curData[self.lastind, 1] - self._seq.data[self.curFrame].dpt.shape[1] // 4,
                             self.curData[self.lastind, 1] + self._seq.data[self.curFrame].dpt.shape[1] // 4])
        self.ax_yz.set_ylim([self.curData[self.lastind, 2] - self._seq.config['cube'][2] // 4,
                             self.curData[self.lastind, 2] + self._seq.config['cube'][2] // 4])
        self.ax_yz.set_xlabel('y')
        self.ax_yz.set_ylabel('z')

        self.update()

    def getCurrentStatus(self, filename_joints, filename_pb, filename_vis):
        pbar = progb.ProgressBar(maxval=len(self._seq.data), widgets=['Loading last status', progb.Percentage(), progb.Bar()])
        pbar.start()
        n_frames_joints = 0
        n_frames_pb = 0
        n_frames_vis = 0
        if os.path.isfile(filename_joints):
            f = open(filename_joints, 'r')
            n_frames_joints = len(f.readlines())
            f.close()

        if os.path.isfile(filename_pb):
            f = open(filename_pb, 'r')
            n_frames_pb = len(f.readlines())
            f.close()

        if os.path.isfile(filename_vis):
            f = open(filename_vis, 'r')
            n_frames_vis = len(f.readlines())
            f.close()

        if not (n_frames_joints == n_frames_pb == n_frames_vis):
            raise EnvironmentError("Inconsistent files (joints: {}, pb: {}, vis: {})!".format(n_frames_joints,
                                                                                              n_frames_pb,
                                                                                              n_frames_vis))
        # load data and forward to next to edit
        cache_str_pb = ''
        with open(filename_pb, "r") as inputfile:
            cache_str_pb = inputfile.readlines()
        cache_str_vis = ''
        with open(filename_vis, "r") as inputfile:
            cache_str_vis = inputfile.readlines()
        cache_str_joints = ''
        with open(filename_joints, "r") as inputfile:
            cache_str_joints = inputfile.readlines()

        for i in xrange(len(self._seq.data)):
            pbar.update(i)
            if len(self.subset_idxs) > 0:
                if i not in self.subset_idxs:
                    continue
            pb, pbp = self.importer.poseBitsFromCache(filename_pb, self._seq.data[i].fileName, cache_str_pb)
            vis = self.importer.visibilityFromCache(filename_vis, self._seq.data[i].fileName, cache_str_vis)
            pose = self.importer.poseFromCache(filename_joints, self._seq.data[i].fileName, cache_str_joints)
            if numpy.allclose(pose, numpy.zeros_like(self.importer.default_gtorig)):
                self.curFrame = i
                break
            else:
                ed = {'vis': vis, 'pb': {'pb': pb, 'pbp': pbp}}
                self._seq.data[i] = self._seq.data[i]._replace(gtorig=pose.reshape((-1, 3)), extraData=ed)

        # redo last pose, it might be set to default and saved
        if self.curFrame > 0:
            if len(self.subset_idxs) > 0:
                if self.subset_idxs.index(self.curFrame) - 1 >= 0:
                    self.curFrame = self.subset_idxs[self.subset_idxs.index(self.curFrame) - 1]
            else:
                self.curFrame -= 1

    def saveCurrent(self):

        self._seq.data[self.curFrame] = self._seq.data[self.curFrame]._replace(gtorig=self.curData)
        if hasattr(self._seq.data[self.curFrame], 'extraData'):
            ed = {'pb': {'pbp': self.curPbP, 'pb': self.curPb}, 'vis': self.curVis}
            self._seq.data[self.curFrame] = self._seq.data[self.curFrame]._replace(extraData=ed)

        if self.filename_joints is not None and self.filename_pb is not None and self.filename_vis is not None:
            self.importer.saveSequenceAnnotations(self._seq, {'joints': self.filename_joints,
                                                              'vis': self.filename_vis,
                                                              'pb': self.filename_pb})

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

    def cb_pb_changed_callback(self):
        checkBox = self.sender()
        checkIdx = None
        for idx, cb in enumerate(self.cb_pb):
            if cb.text() == checkBox.text():
                checkIdx = idx
                break

        if checkBox.isChecked() is True:
            inlist = False
            for idx, cb in enumerate(self.curPbP):
                if str(cb) == checkBox.text():
                    inlist = True
                    break
            if inlist is False:
                self.curPbP.append(self.hc.posebits[checkIdx])
                if self.rb_pb[checkIdx].buttons()[0].isChecked():
                    self.curPb.append(self.hc.posebits[checkIdx])
                elif self.rb_pb[checkIdx].buttons()[1].isChecked():
                    self.curPb.append((self.hc.posebits[checkIdx][1], self.hc.posebits[checkIdx][0]))
        else:
            rmIdx = None
            for idx, cb in enumerate(self.curPbP):
                if str(cb) == checkBox.text():
                    rmIdx = idx
                    break
            if rmIdx is not None:
                del self.curPbP[rmIdx]
                del self.curPb[rmIdx]

    def rb_pb_changed_callback(self):
        radioBut = self.sender()
        opts = [k.strip() for k in str(radioBut.text()).split('<')]
        opt1 = "({}, {})".format(opts[0], opts[1])
        opt2 = "({}, {})".format(opts[1], opts[0])
        checkIdx = None
        for idx, cb in enumerate(self.cb_pb):
            if cb.text() == opt1 or cb.text() == opt2:
                checkIdx = idx
                break

        if self.cb_pb[checkIdx].isChecked() is True:
            inlist = False
            for idx, cb in enumerate(self.curPbP):
                if str(cb) == opt1 or str(cb) == opt2:
                    inlist = True
                    break
            if inlist is False:
                self.curPbP.append(self.hc.posebits[checkIdx])
                if self.rb_pb[checkIdx].buttons()[0].isChecked():
                    self.curPb.append(self.hc.posebits[checkIdx])
                elif self.rb_pb[checkIdx].buttons()[1].isChecked():
                    self.curPb.append((self.hc.posebits[checkIdx][1], self.hc.posebits[checkIdx][0]))
            else:
                # remove and readd to account for changes
                rmIdx = None
                for idx, cb in enumerate(self.curPbP):
                    if str(cb) == opt1 or str(cb) == opt2:
                        rmIdx = idx
                        break
                del self.curPbP[rmIdx]
                del self.curPb[rmIdx]
                self.curPbP.append(self.hc.posebits[checkIdx])
                if self.rb_pb[checkIdx].buttons()[0].isChecked():
                    self.curPb.append(self.hc.posebits[checkIdx])
                elif self.rb_pb[checkIdx].buttons()[1].isChecked():
                    self.curPb.append((self.hc.posebits[checkIdx][1], self.hc.posebits[checkIdx][0]))
        else:
            rmIdx = None
            for idx, cb in enumerate(self.curPbP):
                if str(cb) == opt1 or str(cb) == opt2:
                    rmIdx = idx
                    break
            if rmIdx is not None:
                del self.curPbP[rmIdx]
                del self.curPb[rmIdx]

    def solveButton_callback(self):
        self.solve()

    def nextButton_callback(self):
        self.next()

    def prevButton_callback(self):
        self.prev()

    def keyPressEvent(self, event):
        if self.lastind is None:
            return

        if event.text() == 'q':
            QApplication.instance().quit()
        elif event.text() == 'v':
            # toggle visibility
            if self.lastind in self.curVis:
                self.curVis.remove(self.lastind)
            else:
                self.curVis.append(self.lastind)
            self.showCurrent()
            print 'vis'
        elif event.text() == 'p':
            # previous
            self.prev()
            print 'prev'
        elif event.text() == 'n':
            # next
            self.next()
            print 'next'
        elif event.text() == '3':
            # solve
            self.solve()
            print '3D'
        elif event.text() == 'd':
            self.curData = self.importer.default_gtorig.copy() + self.importer.joint3DToImg(self._seq.data[self.curFrame].com)
            self.showCurrent()
            print 'default'
        elif event.text() == 'a':
            self.curData -= self.curData[self.importer.crop_joint_idx]
            self.curData += self.importer.joint3DToImg(self._seq.data[self.curFrame].com)
            self.showCurrent()
            print 'adjust'
        elif event.text() == 'm':
            self.measureHand()
            print 'measure hand'
        elif event.text() == 'r':
            # reload
            self.panned = False
            self.showCurrent()
            print 'reload'
        elif event.text() == 's':
            # save
            if self.filename_joints is not None and self.filename_pb is not None and self.filename_vis is not None:
                self.importer.saveSequenceAnnotations(self._seq, {'joints': self.filename_joints,
                                                                  'vis': self.filename_vis,
                                                                  'pb': self.filename_pb})
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

    def solve(self):
        import cv2
        from semiautoanno import SemiAutoAnno
        from util.handdetector import HandDetector
        from data.transformations import transformPoint2D

        dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
        hd = HandDetector(dm, self.importer.fx, self.importer.fy, importer=self.importer)
        dpt, M, com = hd.cropArea3D(self.importer.joint3DToImg(self._seq.data[self.curFrame].com), size=self._seq.config['cube'])

        crop = numpy.zeros((self.curData.shape[0], 3), numpy.float32)
        for joint in xrange(crop.shape[0]):
            t = transformPoint2D(self.curData[joint], M)
            crop[joint, 0] = t[0]
            crop[joint, 1] = t[1]
            crop[joint, 2] = self.curData[joint, 2]

        train_data = numpy.zeros((1, 1, dpt.shape[0], dpt.shape[1]), dtype='float32')  # num_imgs,stack_size,rows,cols
        imgD = numpy.asarray(dpt.copy(), 'float32')
        imgD[imgD == 0] = self._seq.data[self.curFrame].com[2] + (self._seq.config['cube'][2] / 2.)
        imgD -= self._seq.data[self.curFrame].com[2]
        imgD /= (self._seq.config['cube'][2] / 2.)
        train_data[0] = imgD

        cae_path = ""
        depth_names = [self.path_prefix+self._seq.data[self.curFrame].fileName]
        li = numpy.asarray([(crop[:, 0:2] - (train_data.shape[3]/2.)) / (train_data.shape[3]/2.)], dtype='float32').clip(-1., 1.)
        train_off3D = numpy.asarray([self._seq.data[self.curFrame].com], dtype='float32')
        train_trans2D = numpy.asarray([numpy.asarray(self._seq.data[self.curFrame].T).transpose()], dtype='float32')
        train_scale = numpy.asarray([self._seq.config['cube'][2]/2.], dtype='float32')
        hc_pm = self.hc.hc_projectionMat()  # create 72 by #Constraints matrix that specifies constant joint length
        boneLength = numpy.asarray(self.hc.boneLength, dtype='float32')
        boneLength /= numpy.asarray([self._seq.config['cube'][2]/2.])[:, None]
        lu_pm = self.hc.lu_projectionMat()  # create 72 by #Constraints matrix that specifies bounds on variable joint length
        boneRange = numpy.asarray(self.hc.boneRanges, dtype='float32')
        boneRange /= numpy.asarray([self._seq.config['cube'][2]/2.])[:, None, None]
        zz_thresh = self.hc.zz_thresh
        zz_pairs = self.hc.zz_pairs
        zz_pairs_v1M, zz_pairs_v2M = self.hc.zz_projectionMat()

        li_visiblemask = numpy.ones((1, self.curData.shape[0]))
        # remove not visible ones
        occluded = numpy.setdiff1d(numpy.arange(li_visiblemask.shape[1]), self.curVis)
        li_visiblemask[0, occluded] = 0

        print self.curVis
        print self.curPb
        print self.curPbP

        eval_params = {'init_method': 'closest',
                       'init_manualrefinement': True,  # True, False
                       'init_offset': 'siftflow',
                       'init_fallback': False,  # True, False
                       'init_incrementalref': True,  # True, False
                       'init_refwithinsequence': True,  # True, False
                       'init_optimize_incHC': True,  # True, False
                       'ref_descriptor': 'hog',
                       'ref_cluster': 'sm_greedy',
                       'ref_fraction': 10.,  # % of samples used as reference at max
                       'ref_threshold': 0.1,  # % of samples used as reference at max
                       'ref_optimization': 'SLSQP',
                       'ref_optimize_incHC': True,  # True, False
                       'joint_eps': 10.,  # mm, visible joints must stay within +/- eps to initialization
                       'joint_off': self.hc.joint_off,  # all joints must be smaller than depth from depth map
                       'eval_initonly': True,  # True, False
                       'eval_refonly': True,  # True, False
                       'optimize_bonelength': False,  # True, False
                       'optimize_Ri': False,  # True, False
                       'global_optimize_incHC': True,  # True, False
                       'global_corr_ref': 'closest',
                       'global_tempconstraints': 'local',  # local, global, none
                       'corr_patch_size': 24,  # px
                       'corr_method': cv2.TM_CCORR_NORMED
                       # cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED
                       }

        msr = SemiAutoAnno(os.path.expanduser('~'), eval_params, train_data, train_off3D, train_trans2D, train_scale,
                           boneLength, boneRange, li, [0], self.importer.getCameraProjection(), cae_path, hc_pm,
                           self.hc.hc_pairs, lu_pm, self.hc.lu_pairs, [self.curPbP], self.hc.posebits,
                           self.hc.pb_thresh, [0], [0], [0],
                           zz_pairs, zz_thresh, zz_pairs_v1M, zz_pairs_v2M, self.hc.finger_tips,
                           1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1, self.importer, self.hc, depth_names,
                           li_visiblemask=li_visiblemask, li_posebits=[self.curPb], normalizeRi=None,
                           hpe=self.hpe, useCache=False, normZeroOne=False, gt3D=None, debugPrint=False)

        # plot 3D
        cur3D = msr.li3D_aug[0]*train_scale+train_off3D
        cur2D = self.importer.joints3DToImg(cur3D)
        cur2Dcrop = numpy.zeros((cur2D.shape[0], 3), numpy.float32)
        for joint in xrange(cur2D.shape[0]):
            t = transformPoint2D(cur2D[joint], M)
            cur2Dcrop[joint, 0] = t[0]
            cur2Dcrop[joint, 1] = t[1]
            cur2Dcrop[joint, 2] = cur2D[joint, 2]
        self.hpe.plotResult(dpt, cur2Dcrop, cur2Dcrop, showGT=False, niceColors=True, visibility=li_visiblemask[0])
        self.hpe.plotResult3D(self._seq.data[self.curFrame].dpt, self._seq.data[self.curFrame].T,
                              cur3D, cur3D, showGT=False, niceColors=True, visibility=li_visiblemask[0])

        # update 3D z-locations
        self.curData[:, 2] = cur3D[:, 2]

        if self.replace_file is not None:
            print "replace anno for {}".format(self.curFrame)
            data = numpy.load(self.replace_file)
            if isinstance(data, numpy.lib.npyio.NpzFile):
                dat = data['arr_1']
                dat[self.curFrame-self.replace_off] = msr.li3D_aug[0].reshape(dat[self.curFrame].shape)
                numpy.savez(self.replace_file, *[data['arr_0'], dat, data['arr_2'], data['arr_3']])
            else:
                data[self.curFrame-self.replace_off] = msr.li3D_aug[0].reshape(data[self.curFrame].shape)
                numpy.save(self.replace_file, data)
            print "saved file {}".format(self.replace_file)

        plt.close()
        del msr
        gc.collect()
        gc.collect()
        gc.collect()

    def measureHand(self):
        cur3D = self.importer.jointsImgTo3D(self.curData)
        print "HC:"
        for i in xrange(len(self.hc.hc_pairs)):
            print str(numpy.sqrt(numpy.square(cur3D[self.hc.hc_pairs[i][0]] - cur3D[self.hc.hc_pairs[i][1]]).sum()))+', '
        print "LU:"
        for i in xrange(len(self.hc.lu_pairs)):
            print str(numpy.sqrt(numpy.square(cur3D[self.hc.lu_pairs[i][0]] - cur3D[self.hc.lu_pairs[i][1]]).sum()))+', '

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

        self.curCorrected = []

        # reset timer
        self.start_time = time.time()

        # previous frame
        prevFrame = self.curFrame
        if len(self.subset_idxs) > 0:
            if self.subset_idxs.index(self.curFrame) - 1 >= 0:
                prevFrame = self.subset_idxs[self.subset_idxs.index(self.curFrame) - 1]
        else:
            if self.curFrame > 0:
                prevFrame = self.curFrame - 1

        if self.replace_file is not None:
            data = numpy.load(self.replace_file)
            if isinstance(data, numpy.lib.npyio.NpzFile):
                dat = data['arr_1'][self.curFrame-self.replace_off].reshape(self._seq.data[self.curFrame].gtorig.shape)*self._seq.config['cube'][2]/2.+self._seq.data[self.curFrame].com
            else:
                dat = data[self.curFrame-self.replace_off].reshape(self._seq.data[self.curFrame].gtorig.shape)*self._seq.config['cube'][2]/2.+self._seq.data[self.curFrame].com
            self.curData = self.importer.joints3DToImg(dat)
        else:
            # if there is no data, use the previous data to initialize
            gtorig = self._seq.data[self.curFrame].gtorig
            if numpy.allclose(gtorig, 0.):
                self.curData = self._seq.data[prevFrame].gtorig.copy()
            else:
                self.curData = gtorig.copy()

        if hasattr(self._seq.data[self.curFrame], 'extraData') and self._seq.data[self.curFrame].extraData:
            if 'pb' in self._seq.data[self.curFrame].extraData.keys():
                self.curPb = self._seq.data[self.curFrame].extraData['pb']['pb']
                self.curPbP = self._seq.data[self.curFrame].extraData['pb']['pbp']
                if len(self.curPb) == 0 and len(self.curPbP) == 0:
                    pass
                    # self.curPb = self._seq.data[prevFrame].extraData['pb']['pb']
                    # self.curPbP = self._seq.data[prevFrame].extraData['pb']['pbp']
            else:
                self.curPb = []
                self.curPbP = []
                for p in range(len(self.hc.posebits)):
                    if abs(self.curData[self.hc.posebits[p][0], 2] - self.curData[self.hc.posebits[p][1], 2]) > self.hc.pb_thresh:
                        if self.curData[self.hc.posebits[p][0], 2] < self.curData[self.hc.posebits[p][1], 2]:
                            self.curPb.append((self.hc.posebits[p][0], self.hc.posebits[p][1]))
                            self.curPbP.append(self.hc.posebits[p])
                        else:
                            self.curPb.append((self.hc.posebits[p][1], self.hc.posebits[p][0]))
                            self.curPbP.append(self.hc.posebits[p])

            if 'vis' in self._seq.data[self.curFrame].extraData.keys():
                self.curVis = self._seq.data[self.curFrame].extraData['vis']
                # default all visible
                if len(self.curVis) == 0:
                    self.curVis = range(0, self.curData.shape[0])
            else:
                dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
                self.curVis = self.importer.visibilityTest(dm, self.curData, 10.)
        else:
            self.curPb = []
            self.curPbP = []
            for p in range(len(self.hc.posebits)):
                if abs(self.curData[self.hc.posebits[p][0], 2] - self.curData[self.hc.posebits[p][1], 2]) > self.hc.pb_thresh:
                    if self.curData[self.hc.posebits[p][0], 2] < self.curData[self.hc.posebits[p][1], 2]:
                        self.curPb.append((self.hc.posebits[p][0], self.hc.posebits[p][1]))
                        self.curPbP.append(self.hc.posebits[p])
                    else:
                        self.curPb.append((self.hc.posebits[p][1], self.hc.posebits[p][0]))
                        self.curPbP.append(self.hc.posebits[p])
            dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
            self.curVis = self.importer.visibilityTest(dm, self.curData, 10.)
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

        self.curCorrected = []

        # reset timer
        self.start_time = time.time()

        if self.replace_file is not None:
            data = numpy.load(self.replace_file)
            if isinstance(data, numpy.lib.npyio.NpzFile):
                dat = data['arr_1'][self.curFrame-self.replace_off].reshape(self._seq.data[self.curFrame].gtorig.shape)*self._seq.config['cube'][2]/2.+self._seq.data[self.curFrame].com
            else:
                dat = data[self.curFrame-self.replace_off].reshape(self._seq.data[self.curFrame].gtorig.shape)*self._seq.config['cube'][2]/2.+self._seq.data[self.curFrame].com
            self.curData = self.importer.joints3DToImg(dat)
        else:
            self.curData = self._seq.data[self.curFrame].gtorig.copy()
        if hasattr(self._seq.data[self.curFrame], 'extraData') and self._seq.data[self.curFrame].extraData:
            if 'pb' in self._seq.data[self.curFrame].extraData.keys():
                self.curPb = self._seq.data[self.curFrame].extraData['pb']['pb']
                self.curPbP = self._seq.data[self.curFrame].extraData['pb']['pbp']
            else:
                self.curPb = []
                self.curPbP = []
                for p in range(len(self.hc.posebits)):
                    if abs(self.curData[self.hc.posebits[p][0], 2] - self.curData[self.hc.posebits[p][1], 2]) > self.hc.pb_thresh:
                        if self.curData[self.hc.posebits[p][0], 2] < self.curData[self.hc.posebits[p][1], 2]:
                            self.curPb.append((self.hc.posebits[p][0], self.hc.posebits[p][1]))
                            self.curPbP.append(self.hc.posebits[p])
                        else:
                            self.curPb.append((self.hc.posebits[p][1], self.hc.posebits[p][0]))
                            self.curPbP.append(self.hc.posebits[p])

            if 'vis' in self._seq.data[self.curFrame].extraData.keys():
                self.curVis = self._seq.data[self.curFrame].extraData['vis']
            else:
                dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
                self.curVis = self.importer.visibilityTest(dm, self.curData, 10.)
        else:
            self.curPb = []
            self.curPbP = []
            for p in range(len(self.hc.posebits)):
                if abs(self.curData[self.hc.posebits[p][0], 2] - self.curData[self.hc.posebits[p][1], 2]) > self.hc.pb_thresh:
                    if self.curData[self.hc.posebits[p][0], 2] < self.curData[self.hc.posebits[p][1], 2]:
                        self.curPb.append((self.hc.posebits[p][0], self.hc.posebits[p][1]))
                        self.curPbP.append(self.hc.posebits[p])
                    else:
                        self.curPb.append((self.hc.posebits[p][1], self.hc.posebits[p][0]))
                        self.curPbP.append(self.hc.posebits[p])
            dm = self.importer.loadDepthMap(self.path_prefix+self._seq.data[self.curFrame].fileName)
            self.curVis = self.importer.visibilityTest(dm, self.curData, 10.)
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
        self.curCorrected.append(self._ind)
        self.curCorrected = numpy.unique(self.curCorrected).tolist()
