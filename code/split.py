import os


path = '/home/wlw/project/split/data'

for id in os.listdir(path):
    #print 'id:', id
    id_path = path + '/' + id
    for filename in os.listdir(id_path):
        #print '**********************'
        #print 'filename:', filename
        year = filename.split('-')[0]
        #print 'year:', year
        year_path = id_path + '/' + year
        if os.path.isdir(year_path):
            pass
        else:
            os.mkdir(year_path)
        month = filename.split('-')[1]
        #print 'month:', month
        month_path = year_path + '/' + month
        if os.path.isdir(month_path):
            pass
        else:
            os.mkdir(month_path)
        oth = filename.split('-')[-1]
        #print 'oth',oth
        day = oth.split('_')[0]
        #print 'day:', day
        day_path = month_path + '/' + day
        if os.path.isdir(day_path):
            pass
        else:
            os.mkdir(day_path)
        hour = oth.split('_')[1]
        #print 'hour:', hour
        hour_path = day_path + '/' + hour
        if os.path.isdir(hour_path):
            pass
        else:
            os.mkdir(hour_path)
        minute = oth.split('_')[2]
        #print 'minute:', minute
        second = oth.split('_')[3]
        #print 'second:', second
        old_path = id_path + '/' +filename
        #print 'old_path:', old_path
        new_name = year + '-' + month +'-' + day + '-' + hour + '-' + minute + '-' + second
        #print 'newname:', new_name
        new_path = hour_path + '/' + new_name
        #print 'newpath:', new_path
        os.rename(old_path, new_path)
