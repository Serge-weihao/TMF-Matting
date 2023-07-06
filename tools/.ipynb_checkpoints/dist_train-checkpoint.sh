#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd /opt/tiger/mmediting
CONFIG=$1
GPUS=$2
WORKDIR=$3
HDFSDIR=$4
SAVENAME=$5
PORT=${PORT:-29507}
hdfs dfs -mkdir -p $HDFSDIR
#cd $THIS_DIR
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work-dir $WORKDIR ${@:6}
cd /opt/tiger/mmediting
zip -r $SAVENAME $WORKDIR
hdfs dfs -put $SAVENAME $HDFSDIR