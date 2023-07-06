import os
import time
#os.system('hdfs dfs -rm -r '+os.environ.get('ARNOLD_OUTPUT')+'/*')
hdfsdir = os.environ.get('ARNOLD_OUTPUT')
print(hdfsdir)
print('hdfs dfs -rm -r  ' +hdfsdir[:-7])
os.system('hdfs dfs -rm -r  ' +hdfsdir[:-7])
os.system('hdfs dfs -mkdir -p ' +hdfsdir)
print('hdfs dfs -mkdir -p ' +hdfsdir[:-6]+'input')
os.system('hdfs dfs -mkdir -p ' +hdfsdir[:-6]+'input')
i = 0
while True:
    os.system('hdfs dfs -put -f  Tensorboard '+hdfsdir)
    os.system('hdfs dfs -mv ' +hdfsdir+'/Tensorboard '+hdfsdir+'/Tensorboard_'+"%05d"%i)
    os.system('hdfs dfs -mv ' +hdfsdir+'/Tensorboard_'+"%05d"%(i-1)+' '+hdfsdir+'/Tensorboard')
    i=i+1
    time.sleep(120)