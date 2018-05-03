#!/usr/bin/env python
#coding:utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse



CLASSES = ('__background__',
            'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}



def vis_detections(image_name, image_path, filesize, height, width, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    #image_path = 'ceshi/'+image_path
    filesize = filesize
    height = height
    width = width
    fig, ax = plt.subplots(figsize=(12, 12))
    
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        path_list=image_path.split('/')[-7:]
        head=' '
        for path in path_list:
           head=os.path.join(head,path)
        print 'image_path',head
        print filesize,height,width,image_name,class_name,bbox[0],bbox[1],bbox[2],bbox[3],score
        print 'outday',outday

        if class_name=='person'or class_name=='car':
            data = str(head)+' '+str(image_name)+' '+str(class_name)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+str(score)+'\n'
            print data
            txtpath = outday+'/data.txt'
            if os.path.exists(txtpath):
                f=open(txtpath,'a')
                f.write(data)
                f.close()
            else:
                f=open(txtpath,'w')
                f.write(data)
                f.close()

            ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

            ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)    
            #plt.axis('off')
            #plt.tight_layout()
            #plt.draw()
            #plt.savefig(outday+'/'+im_name)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            #det_path = '/home/wlw/Downloads/py-faster-rcnn/data/video_pic/' + str(frame_num/25) + '.jpg'
            det_path=outday+'/'+im_name
            cv2.imwrite(det_path, im)
            
        
        else:
            pass

        
	
    
def demo(net, image_name,image_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(image_path)
    print 'im_file',im_file
    im = cv2.imread(im_file)
    im_size = float(os.path.getsize(im_file))
    filesize = float(round((im_size/(1000*1024)),4))
    height, width, num = im.shape

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #print 'hehe'
        vis_detections(image_name, image_path, filesize, height, width, im, cls, dets, thresh=CONF_THRESH)
        #print 'haha'
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    #parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
    #                    default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    '''
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    '''
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    inpath = '/home/wlw/Downloads/py-faster-rcnn/data/pic_data'
    outpath='/home/wlw/Downloads/py-faster-rcnn/data/pic_results'
    for foldername in os.listdir(inpath):
        folderpath = inpath+'/'+foldername
        outfolder=outpath+'/'+foldername
        if os.path.exists(outfolder)==False:
            os.makedirs(outfolder)

        for idname in os.listdir(folderpath):
            idpath=folderpath+'/'+idname
            outid=outfolder+'/'+idname
            if os.path.exists(outid)==False:
                os.makedirs(outid)
            
            for year in os.listdir(idpath):
                yearpath=idpath+'/'+year
                outyear=outid+'/'+year
                if os.path.exists(outyear)==False:
                    os.makedirs(outyear)

                for month in os.listdir(yearpath):
                    monthpath=yearpath+'/'+month
                    outmonth=outyear+'/'+month
                    if os.path.exists(outmonth)==False:
                        os.makedirs(outmonth)

                    for day in os.listdir(monthpath):
                        daypath=monthpath+'/'+day
                        outday=outmonth+'/'+day
                        if os.path.exists(outday)==False:
                            os.makedirs(outday)

                        for time in os.listdir(daypath):
                            timepath=daypath+'/'+time
                            
                            for filename in os.listdir(timepath):
                                im_name=filename
                                im_path=timepath+'/'+im_name

                                print '------------------------------'
                                print 'Demo for data/pic_data/{}/{}/{}/{}/{}/{}/{}'.format(foldername,\
                                idname,year,month,day,time,im_name)
                                demo(net,im_name,im_path)
                                #plt.savefig(outday+'/'+im_name)
            

       
        

            
