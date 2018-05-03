import os

path="/home/wlw/project/cs"

for filename in os.listdir(path):
	print filename
        old_name_path=path+'/'+filename
        print old_name_path
    	new_name=path+'/'+filename.split('.jpg')[0]+'-'+str(1)+'-'+str(2)+'.jpg'
        os.rename(old_name_path,new_name)
    	print new_name
        print new_name.split('/')[-1]
        
