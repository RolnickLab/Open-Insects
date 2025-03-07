# Open-Insects 


## Installing the virtual environment


```
python -m venv ~/<env>

pip install -r requirements.txt

source ~/<env>/bin/activate
```

## Running experiments


For training,

```
PYTHONPATH=$PYTHONPATH
PYTHONPATH=src:$PYTHONPATH
PYTHONPATH=src/OpenOOD:$PYTHONPATH

export PYTHONPATH

python scripts/main.py \
 --config configs/datasets/<dataset config>.yml \
    configs/preprocessors/<preprocessor config>.yml \
    configs/networks/<network config>.yml \
    configs/pipelines/train/baseline.yml \
    --optimizer.num_epochs 120 \
    --optimizer.warmup_epochs 6 \
    --optimizer.lr 0.01 \
    --run_dir <directory to save checkpoints> \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0 

```

For evaluation, 
```
PYTHONPATH=$PYTHONPATH
PYTHONPATH=src:$PYTHONPATH
PYTHONPATH=src/OpenOOD:$PYTHONPATH

export PYTHONPATH

python $CURDIR/scripts/eval_trained.py \
    --config configs/datasets/<dataset config>.yml  \
      configs/datasets/<ood dataset config>.yml \
      configs/preprocessors/base_preprocessor.yml \
      configs/networks/resnet50.yml \
      configs/pipelines/test/test_ood.yml \
      configs/postprocessors/<the OOD detection method>.yml \
    --network.checkpoint <path to the checkpoint>  \
    --dataset.train.batch_size 512 \
    --num_gpus 1 --num_workers 8 \
    --merge_option merge \
    --seed 0 

```
## Citation

If you find our code useful for your research, please consider citing the paper:
```
@misc{chen2025opensetrecognitionnovelspecies,
      title={Open-Set Recognition of Novel Species in Biodiversity Monitoring}, 
      author={Yuyan Chen and Nico Lang and B. Christian Schmidt and Aditya Jain and Yves Basset and Sara Beery and Maxim Larrivée and David Rolnick},
      year={2025},
      eprint={2503.01691},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.01691}, 
}
```


## Acknowledgement

This codebase is built using [OpenOOD] (https://github.com/Jingkang50/OpenOOD/tree/main). We sincerely appreciate their efforts in making this valuable resource publicly available.