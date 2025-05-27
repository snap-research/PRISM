#!/bin/bash


DATASETS=('MovieLens1M') 

echo $DATASET
SEEDS=(123 246 492)
LOSSES=('align')
HIDDEN_DIMS=(64)

for DATASET in "${DATASETS[@]}" ; do
    for SEED in "${SEEDS[@]}" ; do
        for LOSS in "${LOSSES[@]}" ; do

            if [ $LOSS = "BPR"  ]; then
                REGS=("-1")
            elif [ $LOSS = "align"  ]; then
                REGS=("uniformity")
            elif [ $LOSS = "SSM"  ]; then
                REGS=("-1")
            elif [ $LOSS = "MAWU"  ]; then
                REGS=("-1")
            fi

            for REG in  "${REGS[@]}" ; do
                for HIDDEN_DIM in "${HIDDEN_DIMS[@]}" ; do
                    python test.py --dataset $DATASET --model MLP --seed $SEED --loss $LOSS --hidden_dim $HIDDEN_DIM --reg_types $REG --model_save_path "model_chkps"
                done
            done
        done
    done
done