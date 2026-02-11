nohup  torchrun --nproc_per_node=4 train.py --config configs/train_librespeech.yaml --distributed > train.log &
