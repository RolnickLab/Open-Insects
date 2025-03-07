#!/bin/bash
# sh scripts/ood/mixoe/cifar100_train_mixoe.sh

SEED=0
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_oe.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_mixoe.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s${SEED}/best.ckpt \
    --optimizer.lr 0.001 \
    --optimizer.num_epochs 10 \
    --dataset.train.batch_size 128 \
    --dataset.oe.batch_size 128 \
    --seed ${SEED}
