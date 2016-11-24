"""This is the main file for training hand joint classifier on ICVL dataset

Created on 08.08.2014

@author: Markus Oberweger <oberweger@icg.tugraz.at>
"""

from multiprocessing import Pool
import numpy
import scipy
from sklearn.kernel_approximation import SkewedChi2Sampler, RBFSampler
import cv2
import matplotlib
matplotlib.use('Agg')  # plot to file
from semiautoanno import SemiAutoAnno
from util.handconstraints import Blender2HandConstraints
import matplotlib.pyplot as plt
import os
import cPickle
import sys
from data.transformations import transformPoint2D
from data.importers import Blender2Importer
from data.dataset import Blender2Dataset
from util.handpose_evaluation import Blender2HandposeEvaluation


if __name__ == '__main__':

    eval_prefix = 'BLEN_SA'
    if not os.path.exists('./eval/'+eval_prefix+'/'):
        os.makedirs('./eval/'+eval_prefix+'/')

    rng = numpy.random.RandomState(23455)

    print("create data")

    di = Blender2Importer('../data/Blender/')
    Seq1_0 = di.loadSequence('hpseq_loop_mv', camera=0, shuffle=False)
    trainSeqs = [Seq1_0]

    # create training data
    trainDataSet = Blender2Dataset(trainSeqs)
    dat = []
    gt = []
    for seq in trainSeqs:
        d, g = trainDataSet.imgStackDepthOnly(seq.name)
        dat.append(d)
        gt.append(g)
    train_data = numpy.concatenate(dat)
    train_gt3D = numpy.concatenate(gt)

    mb = (train_data.nbytes) / (1024 * 1024)
    print("data size: {}Mb".format(mb))

    imgSizeW = train_data.shape[3]
    imgSizeH = train_data.shape[2]
    nChannels = train_data.shape[1]

    hpe = Blender2HandposeEvaluation([i.gt3Dorig for i in Seq1_0.data], [i.gt3Dorig for i in Seq1_0.data])
    hpe.subfolder += '/'+eval_prefix+'/'

    hc = Blender2HandConstraints([Seq1_0.name])

    # subset of all poses known, e.g. 10% labelled
    # TODO: Set this to [] in order to run reference frame selection
    subset_idxs = []
    # subset_idxs = [16, 21, 26, 29, 45, 49, 52, 54, 58, 104, 108, 114, 138, 144, 148, 170, 175, 178, 210, 214, 217, 231, 237, 249, 252, 259, 264, 283, 287, 296, 307, 345, 370, 381, 384, 386, 405, 412, 423, 429, 436, 458, 465, 469, 490, 498, 505, 526, 530, 533, 537, 546, 553, 576, 607, 612, 624, 631, 657, 667, 669, 673, 685, 697, 704, 735, 742, 751, 765, 781, 784, 789, 793, 801, 805, 816, 820, 827, 830, 874, 886, 888, 893, 896, 899, 911, 923, 934, 962, 969, 983, 1023, 1027, 1029, 1034, 1046, 1054, 1057, 1070, 1075, 1085, 1093, 1098, 1110, 1114, 1134, 1138, 1146, 1173, 1181, 1184, 1188, 1191, 1194, 1208, 1213, 1221, 1224, 1228, 1241, 1248, 1251, 1255, 1262, 1267, 1274, 1286, 1295, 1308, 1312, 1335, 1341, 1349, 1353, 1383, 1386, 1389, 1410, 1414, 1422, 1432, 1449, 1452, 1455, 1465, 1473, 1477, 1489, 1504, 1523, 1532, 1542, 1550, 1552, 1571, 1580, 1586, 1591, 1609, 1613, 1617, 1628, 1632, 1644, 1653, 1656, 1688, 1694, 1695, 1698, 1709, 1713, 1725, 1745, 1752, 1756, 1762, 1772, 1778, 1795, 1812, 1814, 1817, 1830, 1833, 1848, 1853, 1858, 1864, 1869, 1873, 1887, 1892, 1897, 1904, 1927, 1930, 1934, 1937, 1965, 1973, 1978, 1991, 2017, 2028, 2033, 2048, 2055, 2058, 2067, 2074, 2094, 2131, 2137, 2146, 2150, 2166, 2170, 2177, 2185, 2191, 2196, 2203, 2208, 2213, 2222, 2255, 2269, 2273, 2288, 2291, 2298, 2305, 2325, 2331, 2334, 2339, 2343, 2347, 2351, 2372, 2380, 2390, 2394, 2416, 2428, 2434, 2462, 2468, 2484, 2497, 2504, 2509, 2511, 2515, 2529, 2543, 2566, 2572, 2584, 2590, 2609, 2617, 2627, 2631, 2644, 2651, 2654, 2661, 2685, 2687, 2693, 2702, 2737, 2749, 2754, 2763, 2775, 2778, 2790, 2792, 2808, 2813, 2816, 2820, 2829, 2835, 2852, 2856, 2872, 2891, 2898, 2905, 2911, 2942, 2945, 2949, 2952, 2989, 3011, 3015, 3031, 3034, 3037]

    def getSeqIdxForFlatIdx(i):
        nums = numpy.insert(numpy.cumsum(numpy.asarray([len(s.data) for s in trainSeqs])), 0, 0)
        d1 = nums - i
        d1[d1 > 0] = -max([len(s.data) for s in trainSeqs])
        seqidx = numpy.argmax(d1)
        idx = i - nums[seqidx]
        return seqidx, idx

    # mark reference frames
    for i in subset_idxs:
        seqidx, idx = getSeqIdxForFlatIdx(i)
        trainSeqs[seqidx].data[idx] = trainSeqs[seqidx].data[idx]._replace(subSeqName="ref")

    eval_params = {'init_method': 'closest',
                   'init_manualrefinement': True,  # True, False
                   'init_offset': 'siftflow',
                   'init_fallback': False,  # True, False
                   'init_incrementalref': True,  # True, False
                   'init_refwithinsequence': False,  # True, False
                   'init_optimize_incHC': True,  # True, False
                   'init_accuracy_tresh': 10.,  # px
                   'ref_descriptor': 'hog',
                   'ref_cluster': 'sm_greedy',
                   'ref_fraction': 10.,  # % of samples used as reference at max
                   'ref_threshold': 0.08,  # % of samples used as reference at max
                   'ref_optimization': 'SLSQP',
                   'ref_optimize_incHC': True,  # True, False
                   'joint_eps': 15.,  # mm, visible joints must stay within +/- eps to initialization
                   'joint_off': hc.joint_off,  # all joints must be smaller than depth from depth map
                   'eval_initonly': False,  # True, False
                   'eval_refonly': False,  # True, False
                   'optimize_bonelength': False,  # True, False
                   'optimize_Ri': False,  # True, False
                   'global_optimize_incHC': True,  # True, False
                   'global_corr_ref': 'closest',
                   'global_tempconstraints': 'local',  # local, global, none
                   'corr_patch_size': 24,  # px
                   'corr_method': cv2.TM_CCORR_NORMED  # cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED
                   }

    #############################################################################
    cae_path = ""
    depth_names = [ds.fileName for s in trainSeqs for ds in s.data]
    li = numpy.asarray([(ds.gtcrop[:, 0:2] - (train_data.shape[3]/2.)) / (train_data.shape[3]/2.) for s in trainSeqs for ds in s.data], dtype='float32')[subset_idxs].clip(-1., 1.)
    train_off3D = numpy.asarray([ds.com for s in trainSeqs for ds in s.data], dtype='float32')
    train_trans2D = numpy.asarray([numpy.asarray(ds.T).transpose() for s in trainSeqs for ds in s.data], dtype='float32')
    train_scale = numpy.asarray([s.config['cube'][2]/2. for s in trainSeqs], dtype='float32').repeat([len(s.data) for s in trainSeqs])
    hc_pm = hc.hc_projectionMat()  # create 72 by #Constraints matrix that specifies constant joint length
    boneLength = numpy.asarray(hc.boneLength, dtype='float32').reshape((len(trainSeqs), len(hc.hc_pairs)))
    boneLength /= numpy.asarray([s.config['cube'][2]/2. for s in trainSeqs])[:, None]
    lu_pm = hc.lu_projectionMat()  # create 72 by #Constraints matrix that specifies bounds on variable joint length
    boneRange = numpy.asarray(hc.boneRanges, dtype='float32').reshape((len(trainSeqs), len(hc.lu_pairs), 2))
    boneRange /= numpy.asarray([s.config['cube'][2]/2. for s in trainSeqs])[:, None, None]
    zz_thresh = hc.zz_thresh
    zz_pairs = hc.zz_pairs
    zz_pairs_v1M, zz_pairs_v2M = hc.zz_projectionMat()

    nums = numpy.insert(numpy.cumsum(numpy.asarray([len(s.data) for s in trainSeqs])), 0, 0)
    li_visiblemask = numpy.ones((len(subset_idxs), trainSeqs[0].data[0].gtorig.shape[0]))
    for i, sidx in enumerate(subset_idxs):
        seqidx, idx = getSeqIdxForFlatIdx(sidx)
        vis = di.visibilityTest(di.loadDepthMap(trainSeqs[seqidx].data[idx].fileName),
                                trainSeqs[seqidx].data[idx].gtorig, 10.)
        # remove not visible ones
        occluded = numpy.setdiff1d(numpy.arange(li_visiblemask.shape[1]), vis)
        li_visiblemask[i, occluded] = 0

    # li_visiblemask = None

    li_posebits = []
    pb_pairs = []
    for i in range(len(subset_idxs)):
        seqidx, idx = getSeqIdxForFlatIdx(i)
        lip = []
        pip = []
        for p in range(len(hc.posebits)):
            if abs(train_gt3D[subset_idxs[i], hc.posebits[p][0], 2] - train_gt3D[subset_idxs[i], hc.posebits[p][1], 2]) > hc.pb_thresh/(trainSeqs[seqidx].config['cube'][2]/2.):
                if train_gt3D[subset_idxs[i], hc.posebits[p][0], 2] < train_gt3D[subset_idxs[i], hc.posebits[p][1], 2]:
                    lip.append((hc.posebits[p][0], hc.posebits[p][1]))
                    pip.append(hc.posebits[p])
                else:
                    lip.append((hc.posebits[p][1], hc.posebits[p][0]))
                    pip.append(hc.posebits[p])
        li_posebits.append(lip)
        pb_pairs.append(pip)

    # li_posebits = None
    # pb_pairs = []

    #############################################################
    # simple run
    lambdaW = 1e-5
    lambdaM = 1e0
    lambdaR = 1e0
    lambdaP = 1e2
    lambdaMu = 1e-2
    lambdaTh = 1e0
    muLagr = 1e2
    ref_lambdaP = 1e1
    ref_muLagr = 1e1
    init_lambdaP = 1e0
    init_muLagr = 1e1
    msr = SemiAutoAnno('./eval/' + eval_prefix, eval_params, train_data, train_off3D, train_trans2D, train_scale,
                       boneLength, boneRange, li, subset_idxs, di.getCameraProjection(), cae_path, hc_pm,
                       hc.hc_pairs, lu_pm, hc.lu_pairs, pb_pairs, hc.posebits, hc.pb_thresh,
                       hc.getTemporalBreaks(), hc.getHCBreaks(), hc.getSequenceBreaks(),
                       zz_pairs, zz_thresh, zz_pairs_v1M, zz_pairs_v2M, hc.finger_tips,
                       lambdaW, lambdaM, lambdaP, lambdaR, lambdaMu, lambdaTh, muLagr,
                       ref_lambdaP, ref_muLagr, init_lambdaP, init_muLagr, di, hc, depth_names,
                       li_visiblemask=li_visiblemask, li_posebits=li_posebits, normalizeRi=None,
                       useCache=True, normZeroOne=False, gt3D=train_gt3D, hpe=hpe, debugPrint=False)

    # Test initialization
    jts = msr.li3D_aug
    gt3D = [j.gt3Dorig for s in trainSeqs for j in s.data]
    joints = []
    for i in range(jts.shape[0]):
        seqidx, idx = getSeqIdxForFlatIdx(subset_idxs[i])
        joints.append(jts[i].reshape(trainSeqs[seqidx].data[0].gt3Dorig.shape)*(trainSeqs[seqidx].config['cube'][2]/2.) + trainSeqs[seqidx].data[idx].com)

    hpe = Blender2HandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    hpe.subfolder += '/'+eval_prefix+'/'
    print("Initialization:")
    print("Subset samples: {}".format(len(subset_idxs)))
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe.getMeanError(), hpe.getMaxError(), hpe.getMedianError()))
    print("{}".format([hpe.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    hpe_vis = Blender2HandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], joints)
    hpe_vis.subfolder += '/'+eval_prefix+'/'
    hpe_vis.maskVisibility(numpy.repeat(li_visiblemask[:, :, None], 3, axis=2))
    print("Only visible joints:")
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe_vis.getMeanError(), hpe_vis.getMaxError(), hpe_vis.getMedianError()))
    print("{}".format([hpe_vis.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe_vis.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    # Tracking
    theta = msr.fitTracking(num_iter=None, useLagrange=False)
    numpy.savez('./eval/'+eval_prefix+'/params_tracking.npz', *theta)

    print("Testing ...")
    
    # evaluate optimization result
    gt3D = [j.gt3Dorig for s in trainSeqs for j in s.data]
    joints = []
    for i in range(train_data.shape[0]):
        seqidx, idx = getSeqIdxForFlatIdx(i)
        joints.append(theta[1][i].reshape(trainSeqs[seqidx].data[0].gt3Dorig.shape)*(trainSeqs[seqidx].config['cube'][2]/2.) + trainSeqs[seqidx].data[idx].com)

    hpe_ref = Blender2HandposeEvaluation(numpy.asarray(gt3D)[subset_idxs], numpy.asarray(joints)[subset_idxs])
    hpe_ref.subfolder += '/'+eval_prefix+'/'
    print("Reference samples: {}".format(len(subset_idxs)))
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe_ref.getMeanError(), hpe_ref.getMaxError(), hpe_ref.getMedianError()))
    print("{}".format([hpe_ref.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe_ref.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    joints = []
    for i in range(train_data.shape[0]):
        seqidx, idx = getSeqIdxForFlatIdx(i)
        joints.append(theta[2][i].reshape(trainSeqs[seqidx].data[0].gt3Dorig.shape)*(trainSeqs[seqidx].config['cube'][2]/2.) + trainSeqs[seqidx].data[idx].com)

    hpe_init = Blender2HandposeEvaluation(gt3D, joints)
    hpe_init.subfolder += '/'+eval_prefix+'/'
    print("Train initialization: {}".format(train_data.shape[0]))
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe_init.getMeanError(), hpe_init.getMaxError(), hpe_init.getMedianError()))
    print("{}".format([hpe_init.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe_init.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    joints = []
    for i in range(train_data.shape[0]):
        seqidx, idx = getSeqIdxForFlatIdx(i)
        joints.append(theta[1][i].reshape(trainSeqs[seqidx].data[0].gt3Dorig.shape)*(trainSeqs[seqidx].config['cube'][2]/2.) + trainSeqs[seqidx].data[idx].com)

    hpe_full = Blender2HandposeEvaluation(gt3D, joints)
    hpe_full.subfolder += '/'+eval_prefix+'/'
    print("Train samples: {}".format(train_data.shape[0]))
    print("Mean error: {}mm, max error: {}mm, median error: {}mm".format(hpe_full.getMeanError(), hpe_full.getMaxError(), hpe_full.getMedianError()))
    print("{}".format([hpe_full.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe_full.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    # save final joint annotations, in original scaling
    joints2D = numpy.zeros_like(joints)
    for i in xrange(len(joints)):
        joints2D[i] = di.joints3DToImg(joints[i])
    numpy.savez('./eval/'+eval_prefix+'/new_joints.npz', new2D=numpy.asarray(joints2D), new3D=numpy.asarray(joints))

    ind=0
    for i in trainSeqs[0].data:
        jt = joints[ind]
        jtI = di.joints3DToImg(jt)
        for joint in range(jt.shape[0]):
            t=transformPoint2D(jtI[joint],i.T)
            jtI[joint,0] = t[0]
            jtI[joint,1] = t[1]
        hpe_full.plotResult(i.dpt, i.gtcrop, jtI,"{}_optimized_{}".format(eval_prefix, ind))
        if ind < 100:
            ind += 1
        else:
            break

    hpe_init.saveVideo('P0_init', trainSeqs[0], di, fullFrame=True, plotFrameNumbers=True)
    hpe_full.saveVideo('P0', trainSeqs[0], di, fullFrame=True, plotFrameNumbers=True)
    hpe_init.saveVideo3D('P0_init', trainSeqs[0])
    hpe_full.saveVideo3D('P0', trainSeqs[0])

