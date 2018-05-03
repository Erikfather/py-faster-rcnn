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




def target_recognition(net, image):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = image

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])


    # Detect each class
    recognition_result = []
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
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        else:
            for i in inds:
                #if cls != 'person':
                #    break
                bbox = dets[i, :4]
                score = dets[i, -1]
                data = [cls, bbox[0], bbox[1], bbox[2], bbox[3], score]
                recognition_result.append(data)
    return recognition_result



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

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

    caffe.set_mode_cpu()
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    inpath = '/home/wlw/project/bad/'
    outpath='/home/wlw/project/pic_result'
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
                                path_list=im_path.split("/")[-7:]
                                path=' '
                                for pathname in path_list:
                                    path=os.path.join(path,pathname)
                                print '------------------------------'
                                print 'Demo for data/pic_data/{}/{}/{}/{}/{}/{}/{}'.format(foldername,idname,year,month,day,time,im_name)
                                frame = cv2.imread(im_path)
                                if str(frame) == 'None':
                                    continue
                                recognition_result = target_recognition(net, frame)
                                target_num = 0
                                for target in recognition_result:
                                    #targrt_num = target_num + 1
                                    class_name = target[0]
                                    bbox = [target[1], target[2], target[3], target[4]]
                                    score = target[5]
                                    print class_name, bbox, score
                                    if class_name=='car'or class_name=='person':
                                        target_num = target_num + 1
                                        print str(target_num)
                                        data = str(path)+' '+str(im_name)+' '+str(class_name)+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(score)+'\n'
                                        txtpath = outday+'/data.txt'
                                        if os.path.exists(txtpath):
                                            f=open(txtpath,'a')
                                            f.write(data)
                                            f.close()
                                        else:
                                            f=open(txtpath,'w')
                                            f.write(data)
                                            f.close()
                                        if class_name=='car':
                                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                                        if class_name=='person':
                                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                                        #det_path=outputfolder+'/'+str(frame_num/25) + '.jpg'
                                        det_path=outday+'/'+im_name
                                        cv2.imwrite(det_path, frame)
                                
                                        txtpath2 = outday+'/num.txt'
                                
                                if target_num>0:
                                    data2=str(path)+' '+str(im_name)+' '+str(target_num)
				    lst.append(data2)
                            print 'ls1', lst
                            lst.sort(key=operator.itemgetter(-1),reverse=True)
                            print 'ls2', lst
                            if len(lst) < 3:
                                for i in lst:
                                    if os.path.exists(txtpath2):
                                        f=open(txtpath2,'a')
                                        f.write(i)
                                        f.close()
                                    else:                                
                                        f=open(txtpath2,'w')
                                        f.write(i)
                                        f.close()

                            else: 
                                data3=str(lst[0])+'\n'+str(lst[1])+'\n'+str(lst[2])+'\n'
                                                                   
                                if os.path.exists(txtpath2):
                                    f=open(txtpath2,'a')
                                    f.write(data3)
                                    f.close()
                                else:
                                    f=open(txtpath2,'w')
                                    f.write(data3)
                                    f.close()

