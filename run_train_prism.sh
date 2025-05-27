#!/bin/bash
DATASET=$1
MODEL=$2
EPOCHS=$3
LOSS=$4
REG=$5
GAMMA=$6
WD=$7
LR=$8
PATIENCE=$9
DEGREE_INIT_STR=${10}
DATA_PATH=${11}

DATASET='MovieLens1M'
MODEL='MLP'
EPOCHS=1000
LOSS='align'
REG='uniformity'
GAMMA='1.0'
WD='0.0'
LR='0.001'
PATIENCE='10'
DEGREE_INIT_STR='1.0'
DATA_PATH='model_chkps'

echo $DATASET

SEEDS=(123 246 492)
HIDDEN_DIMS=(64) 


if [ $MODEL = "MLP" ]; then
    NUM_LAYERS=(0 2)
elif [ $MODEL = "LGConv" ]; then
    NUM_LAYERS=(3)
fi

if [ $LOSS = "BPR"  ]; then
    BATCH_SIZE=16384
    NEG_RATIOS=(1)
elif [ $LOSS = "SSM"  ]; then
    BATCH_SIZE=16384
    NEG_RATIOS=(20) 
elif [ $LOSS = "align"  ]; then
    BATCH_SIZE=4096
    NEG_RATIOS=(0)
elif [ $LOSS = "MAWU"  ]; then
    BATCH_SIZE=4096
    NEG_RATIOS=(0)
fi

for SEED in "${SEEDS[@]}" ; do
    for HIDDEN_DIM in "${HIDDEN_DIMS[@]}" ; do
        for NUM_LAYER in "${NUM_LAYERS[@]}" ; do
            for NEG_RATIO in "${NEG_RATIOS[@]}" ; do
                python train.py --model $MODEL --dataset $DATASET --epochs $EPOCHS --lr $LR \
                                --seed $SEED --loss $LOSS --hidden_dim $HIDDEN_DIM --num_layers $NUM_LAYER --weight_decay $WD \
                                --batch_size $BATCH_SIZE --neg_ratio $NEG_RATIO --reg_types $REG \
                                --gamma_vals $GAMMA --patience $PATIENCE --degree_init True --overwrite True \
                                --degree_init_str $DEGREE_INIT_STR --model_save_path $DATA_PATH
                wait
            done
        done
    done
done
