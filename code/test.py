import os

suoluepath='/home/wlw/project/SL_result/huaihai/901/2017/06/'

for day in os.listdir(suoluepath):
    if day != 'suo_month':
        print 'day', day


day_pic = '2017-05-06-09-00-01.jpg'
day_pic = day_pic.split('-')
print day_pic


inpath = '/home/wlw/project/test/test_month'
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

    for idname in os.listdir(folderpath):
        idpath=folderpath+'/'+idname
        outid=outfolder+'/'+idname
        suoid=suofolder+'/'+idname
        if os.path.exists(outid)==False:
            os.makedirs(outid)
        if os.path.exists(suoid)==False:
            os.makedirs(suoid)
            
        for year in os.listdir(idpath):
            yearpath=idpath+'/'+year
            outyear=outid+'/'+year
            suoyear=suoid+'/'+year
            if os.path.exists(outyear)==False:
                os.makedirs(outyear)
            if os.path.exists(suoyear)==False:
                os.makedirs(suoyear)

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
                        print 'filenames', len(filenames)
                        if len(filenames) == 0:
                            print '11111111111111'
                            continue
                        first_pic_path = timepath + '/' + filenames[0]
