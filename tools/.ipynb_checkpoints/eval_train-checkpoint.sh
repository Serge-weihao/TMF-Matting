#!/bin/bash
mkdir -p ~/.encoding/models
cd ~/.encoding/models
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/user/yudongdong/encoding-models/*.pth .
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd /opt/tiger/mmediting
hdfs dfs -get hdfs://haruna//home/byte_arnold_lq_vc/jiangweihao.serge/init_matting/*.pth
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_vc/jiangweihao.serge/datasets/Adobegca.zip
unzip Adobegca.zip
CONFIG=$1
GPUS=$2
WORKDIR=$3
HDFSDIR=$4
SAVENAME=$5
PORT=${6:-29507}
hdfs dfs -mkdir -p $HDFSDIR
#cd $THIS_DIR
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work-dir $WORKDIR ${@:7}
cd /opt/tiger/mmediting
zip -r $SAVENAME $WORKDIR
hdfs dfs -put $SAVENAME $HDFSDIR
dir=$(ls $WORKDIR/*.pth)
for i in $dir
do
echo 'start '$i' eval'
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $i --launcher pytorch
echo 'done '$i' eval'
done