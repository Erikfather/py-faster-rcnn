#coding=utf-8

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
import operator
import shutil

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

    inpath = '/home/wlw/project/bad'
    outpath='/home/wlw/project/pic_result'
    suoluepath='/home/wlw/project/SL_result'
    for foldername in os.listdir(inpath):
        folderpath = inpath+'/'+foldername
        outfolder=outpath+'/'+foldername
        suofolder=suoluepath+'/'+foldername
        if os.path.exists(outfolder)==False:
            os.makedirs(outfolder)
        if os.path.exists(suofolder)==False:
            os.makedirs(suofolder)
   
        lst_id = []
        for idname in os.listdir(folderpath):
            idpath=folderpath+'/'+idname
            outid=outfolder+'/'+idname
            suoid=suofolder+'/'+idname
            if os.path.exists(outid)==False:
                os.makedirs(outid)
            if os.path.exists(suoid)==False:
                os.makedirs(suoid)
            
            lst_year = []
            for year in os.listdir(idpath):
                yearpath=idpath+'/'+year
                outyear=outid+'/'+year
                suoyear=suoid+'/'+year
                if os.path.exists(outyear)==False:
                    os.makedirs(outyear)
                if os.path.exists(suoyear)==False:
                    os.makedirs(suoyear)
                
                lst_month = []
                for month in os.listdir(yearpath):
                    monthpath=yearpath+'/'+month
                    outmonth=outyear+'/'+month
                    suomonth=suoyear+'/'+month
                    if os.path.exists(outmonth)==False:
                        os.makedirs(outmonth)
                    if os.path.exists(suomonth)==False:
                        os.makedirs(suomonth)
                    
                    lst_day = []
                    for day in os.listdir(monthpath):
                        daypath=monthpath+'/'+day
                        outday=outmonth+'/'+day
                        suoday=suomonth+'/'+day
                        if os.path.exists(outday)==False:
                            os.makedirs(outday)
                        if os.path.exists(suoday)==False:
                            os.makedirs(suoday)

                        lst=[]
                        for time in os.listdir(daypath):
                            timepath=daypath+'/'+time
                            filenames = os.listdir(timepath)
                            filenames.sort()
                            print 'filenames', filenames
                            if len(filenames) == 0:
                                continue

                            #保存第一张图片
                            first_pic_path = timepath + '/' + filenames[0]
                            #print 'first_pic_path:', first_pic_path
                            first_pic = cv2.imread(first_pic_path)
                            first_pic_outpath = outday + '/' + filenames[0]
                            #print first_pic_outpath
                            cv2.imwrite(first_pic_outpath, first_pic)

                            fgbg = cv2.createBackgroundSubtractorKNN()
                            valid_thresh = 0.026
                            
                            for filename in filenames:
           
                                print filename
                                target_pixel_num = 0
                                filepath = timepath + '/' + filename
                                frame = cv2.imread(filepath)
                                if str(frame) == 'None':
                                    continue
                                fgmask = fgbg.apply(frame)
                                (h, w) = fgmask.shape[:2]
                                kernel = np.ones((3, 3), np.uint8)
                                fgmask = cv2.erode(fgmask, kernel, iterations=1)
                                #cv2.imshow('fgmask', fgmask)
                                #if cv2.waitKey(1) & 0xFF == ord('q'):
                                #     break
                                for line in fgmask:
                                    line = list(line)
                                    target_pixel_num += line.count(255)
                                #print 'num:', target_pixel_num
                                #print float(target_pixel_num) / (w * h)
                                #if float(target_pixel_num) / (w * h) > valid_thresh and float(target_pixel_num) / (w * h) < 0.9:
                                #    det_path = outday + '/' + filename
                                #    cv2.imwrite(det_path, frame)

                                #画变化部分包围框
                                if float(target_pixel_num) / (w * h) > valid_thresh and float(target_pixel_num) / (w * h) < 0.9:
                                    fgmask = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)[1]
                                    _, cnts, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    for cnt in cnts:
                                            (x, y, w, h) = cv2.boundingRect(cnt)
                                            area = cv2.contourArea(cnt)
                                            #print 'area', area
                                            if area > 40000:
                                                cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
                                    det_path = outday + '/' + filename
                                    cv2.imwrite(det_path, frame)


                                im_name=filename
                                im_path=timepath+'/'+im_name
                                path_list=im_path.split("/")[-7:]
                                path=''
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
                                
                                        #txtpath2 = suoday+'/num.txt'
                                
                                if target_num>0:
                                    data2=str(path)+' '+str(im_name)+' '+str(target_num)
				    lst.append(data2)
                                print 'lst', lst
                        ''' 
                        #day缩略图
                        print 'ls1', lst
                        lst.sort(key=operator.itemgetter(-1),reverse=True)
                        print 'ls2', lst
                        suolue_pic=suoday+'/suo_pic'
                        #suolue_pic=suoday
                        if os.path.exists(suolue_pic)==False:
                            os.makedirs(suolue_pic)
                        if len(lst) < 3 and len(lst) > 0:
                            for i in lst:
                                i_s = i.split(' ')
                                detect_pic_path = outday + '/' + i_s[1]
                                new_path = suolue_pic + '/' + i_s[1]
                                shutil.copyfile(detect_pic_path, new_path)
                                #print 'dp', detect_pic_path
                                #print 'np', new_path
                                #if os.path.exists(txtpath2):
                                #    f=open(txtpath2,'a')
                                #    f.write(i)
                                #    f.close()
                                #else:                           
                                #    f=open(txtpath2,'w')
                                #    f.write(i)
                                #    f.close()

                        if len(lst)>=3: 
                            data3=str(lst[0])+'\n'+str(lst[1])+'\n'+str(lst[2])
                            data3 = data3.split('\n')
                            for data in data3:
                                print 'data', data
                                data_s = data.split(' ')
                                detect_pic_path = outday + '/' + data_s[1]
                                new_path = suolue_pic + '/' + data_s[1]
                                #print 'dp', detect_pic_path
                                #print 'np', new_path
                                shutil.copyfile(detect_pic_path, new_path)
                                                                 
                                #if os.path.exists(txtpath2):

                                #    f=open(txtpath2,'a')
                                #    f.write(data + '\n')
                                #    f.close()
                                #else:
                                #    f=open(txtpath2,'w')
                                #    f.write(data + '\n')
                                #    f.close()
                     
                        else:
                            continue 

                    #month缩略图
                    #print 'suomonth', suomonth
                    suolue_month = suomonth+'/suo_month'
                    if os.path.exists(suolue_month)==False:
                        os.makedirs(suolue_month)
                    for day in os.listdir(suomonth):
                        if day != 'suo_month':
                            #print 'day', day
                            day = suomonth + '/' + day + '/suo_pic'
                            for pic in os.listdir(day):
                                lst_day.append(pic)
                    #print 'lst_day', lst_day
                    lst_day.sort()
                    print 'sort lst_day', lst_day
                    if len(lst_day) < 3 and len(lst_day) > 0:
                        #print '1111111111111'
                        for day_pic in lst_day:
                            temp = day_pic.split('-')
                            old_path = suoid + '/' + temp[0] + '/' + temp[1] + '/' + temp[2] + '/suo_pic/' + day_pic
                            new_path = suolue_month + '/' + day_pic
                            #print 'old_path', old_path
                            #print 'new_path', new_path
                            shutil.copyfile(old_path, new_path)
                    if len(lst_day) >= 3:
                        #print '222222222222222'
                        pic_1 = lst_day[0]
                        temp = pic_1.split('-')
                        old_path = suoid + '/' + temp[0] + '/' + temp[1] + '/' + temp[2] + '/suo_pic/' + pic_1
                        new_path = suolue_month + '/' + pic_1
                        #print 'pic_1!!!!!!!!!!!!!!!!!!!!!!!!!'
                        #print 'old_path', old_path
                        #print 'new_path', new_path
                        shutil.copyfile(old_path, new_path)
                        pic_2 = lst_day[int(len(lst_day)/2)]
                        temp = pic_2.split('-')
                        old_path = suoid + '/' + temp[0] + '/' + temp[1] + '/' + temp[2] + '/suo_pic/' + pic_2
                        new_path = suolue_month + '/' + pic_2
                        #print 'pic_2!!!!!!!!!!!!!!!!!!!!!!!!!'
                        #print 'old_path', old_path
                        #print 'new_path', new_path
                        shutil.copyfile(old_path, new_path)
                        pic_3 = lst_day[-1]
                        temp = pic_3.split('-')
                        old_path = suoid + '/' + temp[0] + '/' + temp[1] + '/' + temp[2] + '/suo_pic/' + pic_3
                        new_path = suolue_month + '/' + pic_3
                        #print 'pic_3!!!!!!!!!!!!!!!!!!!!!!!!!'
                        #print 'old_path', old_path
                        #print 'new_path', new_path
                        shutil.copyfile(old_path, new_path)
                    else:
                        continue
                         
                        



            #year缩略图
            suolue_year = suoyear+'/suo_year'
            if os.path.exists(suolue_year)==False:
                os.makedirs(suolue_year)
                for month in os.listdir(suoyear):
                    if month != 'suo_year':
                        #print 'day', day
                        month = suoyear + '/' + month + '/suo_month'
                        for pic in os.listdir(month):
                            lst_month.append(pic)
                #print 'lst_day', lst_day
                lst_month.sort()
                print 'sort lst_month', lst_month
                if len(lst_month) < 3 and len(lst_month) > 0:
                    #print '1111111111111'
                    for month_pic in lst_month:
                        temp = month_pic.split('-')
                        old_path = suoid + '/' + temp[0] + '/' + temp[1] + '/suo_month/' + month_pic
                        new_path = suolue_year + '/' + month_pic
                        #print 'old_path', old_path
                        #print 'new_path', new_path
                        shutil.copyfile(old_path, new_path)
                if len(lst_month) >= 3:
                    #print '222222222222222'
                    pic_1 = lst_month[0]
                    temp = pic_1.split('-')
                    old_path = suoid + '/' + temp[0] + '/' + temp[1] + '/suo_month/' + pic_1
                    new_path = suolue_year + '/' + pic_1
                    #print 'pic_1!!!!!!!!!!!!!!!!!!!!!!!!!'
                    #print 'old_path', old_path
                    #print 'new_path', new_path
                    shutil.copyfile(old_path, new_path)
                    pic_2 = lst_month[int(len(lst_month)/2)]
                    temp = pic_2.split('-')
                    old_path = suoid + '/' + temp[0] + '/' + temp[1] + '/suo_month/' + pic_2
                    new_path = suolue_year + '/' + pic_2
                    #print 'pic_2!!!!!!!!!!!!!!!!!!!!!!!!!'
                    #print 'old_path', old_path
                    #print 'new_path', new_path
                    shutil.copyfile(old_path, new_path)
                    pic_3 = lst_month[-1]
                    temp = pic_3.split('-')
                    old_path = suoid + '/' + temp[0] + '/' + temp[1] + '/suo_month/' + pic_3
                    new_path = suolue_year + '/' + pic_3
                    #print 'pic_3!!!!!!!!!!!!!!!!!!!!!!!!!'
                    #print 'old_path', old_path
                    #print 'new_path', new_path
                    shutil.copyfile(old_path, new_path)
                else:
                    continue
        '''
        if os.path.exists(folderpath):
            shutil.rmtree(folderpath)
        else:
            pass
        #id缩略图


    #locate缩略图


