"""This is the main file for annotation of depth files

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

import getopt
import numpy
import os
import sys
import pwd
from PyQt4.QtGui import QApplication
from data.importers import Blender2Importer
from util.handconstraints import Blender2HandConstraints
from util.handpose_evaluation import Blender2HandposeEvaluation
from util.interactivedatasetlabeling import InteractiveDatasetLabeling

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2016, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


if __name__ == '__main__':
    person = None
    username = None
    start_idx = None
    try:
        opts, args = getopt.getopt(sys.argv, "hs:p:u:", ["person=", "start_idx=", "username="])
    except getopt.GetoptError:
        print 'main_labeling_pose.py -p <person> -s <start_idx> -u <username>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print 'main_labeling_pose.py -p <person> -s <start_idx> -u <username>'
            sys.exit()
        elif opt in ("-p", "--person"):
            person = arg.lower()
        elif opt in ("-u", "--username"):
            username = arg.lower()
        elif opt in ("-s", "--start_idx"):
            start_idx = int(arg)

    if person is None:
        print 'Person must be specified: -p <person> or enter now:'
        person = raw_input().lower()
        if len(person.strip()) == 0:
            sys.exit(2)
    else:
        print 'Person is {}'.format(person)

    if username is None:
        print 'Username must be specified: -u <username> or enter now:'
        username = raw_input().lower()
        if len(username.strip()) == 0:
            sys.exit(2)
    else:
        print 'Username is {}'.format(username)

    if start_idx is None:
        print 'Start frame index can be specified: -s <start_idx> or enter now:'
        start_idx = raw_input().lower()
        if len(start_idx.strip()) == 0:
            start_idx = 0
        else:
            start_idx = int(start_idx)
    else:
        print 'Start frame index is {}'.format(start_idx)

    rng = numpy.random.RandomState(23455)

    # subset to label
    subset_idxs = []

    if person == 'hpseq_loop_mv':
        di = Blender2Importer('../data/Blender/', useCache=False)
        Seq2 = di.loadSequence(person, camera=0, shuffle=False)
        hc = Blender2HandConstraints([Seq2.name])
        hpe = Blender2HandposeEvaluation([j.gt3Dorig for j in Seq2.data], [j.gt3Dorig for j in Seq2.data])
        for idx, seq in enumerate(Seq2.data):
            ed = {'vis': [], 'pb': {'pb': [], 'pbp': []}}
            Seq2.data[idx] = seq._replace(gtorig=numpy.zeros_like(seq.gtorig), extraData=ed)

        # common subset for all
        subset_idxs = [16, 21, 26, 29, 45, 49, 52, 54, 58, 104, 108, 114, 138, 144, 148, 170, 175, 178, 210, 214, 217, 231, 237, 249, 252, 259, 264, 283, 287, 296, 307, 345, 370, 381, 384, 386, 405, 412, 423, 429, 436, 458, 465, 469, 490, 498, 505, 526, 530, 533, 537, 546, 553, 576, 607, 612, 624, 631, 657, 667, 669, 673, 685, 697, 704, 735, 742, 751, 765, 781, 784, 789, 793, 801, 805, 816, 820, 827, 830, 874, 886, 888, 893, 896, 899, 911, 923, 934, 962, 969, 983, 1023, 1027, 1029, 1034, 1046, 1054, 1057, 1070, 1075, 1085, 1093, 1098, 1110, 1114, 1134, 1138, 1146, 1173, 1181, 1184, 1188, 1191, 1194, 1208, 1213, 1221, 1224, 1228, 1241, 1248, 1251, 1255, 1262, 1267, 1274, 1286, 1295, 1308, 1312, 1335, 1341, 1349, 1353, 1383, 1386, 1389, 1410, 1414, 1422, 1432, 1449, 1452, 1455, 1465, 1473, 1477, 1489, 1504, 1523, 1532, 1542, 1550, 1552, 1571, 1580, 1586, 1591, 1609, 1613, 1617, 1628, 1632, 1644, 1653, 1656, 1688, 1694, 1695, 1698, 1709, 1713, 1725, 1745, 1752, 1756, 1762, 1772, 1778, 1795, 1812, 1814, 1817, 1830, 1833, 1848, 1853, 1858, 1864, 1869, 1873, 1887, 1892, 1897, 1904, 1927, 1930, 1934, 1937, 1965, 1973, 1978, 1991, 2017, 2028, 2033, 2048, 2055, 2058, 2067, 2074, 2094, 2131, 2137, 2146, 2150, 2166, 2170, 2177, 2185, 2191, 2196, 2203, 2208, 2213, 2222, 2255, 2269, 2273, 2288, 2291, 2298, 2305, 2325, 2331, 2334, 2339, 2343, 2347, 2351, 2372, 2380, 2390, 2394, 2416, 2428, 2434, 2462, 2468, 2484, 2497, 2504, 2509, 2511, 2515, 2529, 2543, 2566, 2572, 2584, 2590, 2609, 2617, 2627, 2631, 2644, 2651, 2654, 2661, 2685, 2687, 2693, 2702, 2737, 2749, 2754, 2763, 2775, 2778, 2790, 2792, 2808, 2813, 2816, 2820, 2829, 2835, 2852, 2856, 2872, 2891, 2898, 2905, 2911, 2942, 2945, 2949, 2952, 2989, 3011, 3015, 3031, 3034, 3037]
    else:
        raise NotImplementedError("")

    replace_off = 0
    replace_file = None  # './params_tracking.npz'

    output_path = di.basepath

    filename_joints = output_path+person+'/joint_'+username+'.txt'
    filename_pb = output_path+person+'/pb_'+username+'.txt'
    filename_vis = output_path+person+'/vis_'+username+'.txt'
    filename_log = output_path+person+'/annotool_log_'+username+'.txt'

    # create empty file
    if not os.path.exists(filename_joints):
        annoFile = open(filename_joints, "w")
        annoFile.close()
    else:
        bak = filename_joints+'.bak'
        i = 0
        while os.path.exists(bak):
            bak = filename_joints+'.bak.{}'.format(i)
            i += 1
        os.popen('cp '+filename_joints+' '+bak)

    if not os.path.exists(filename_pb):
        annoFile = open(filename_pb, "w")
        annoFile.close()
    else:
        bak = filename_pb+'.bak'
        i = 0
        while os.path.exists(bak):
            bak = filename_pb+'.bak.{}'.format(i)
            i += 1
        os.popen('cp '+filename_pb+' '+bak)

    if not os.path.exists(filename_vis):
        annoFile = open(filename_vis, "w")
        annoFile.close()
    else:
        bak = filename_vis+'.bak'
        i = 0
        while os.path.exists(bak):
            bak = filename_vis+'.bak.{}'.format(i)
            i += 1
        os.popen('cp '+filename_vis+' '+bak)

    app = QApplication(sys.argv)
    browser = InteractiveDatasetLabeling(Seq2, hpe, di, hc, filename_joints, filename_pb, filename_vis, filename_log,
                                         subset_idxs, start_idx, replace_file, replace_off)
    browser.show()
    app.exec_()
    print browser.curData, browser.curVis, browser.curPb, browser.curPbP, browser.curCorrected

