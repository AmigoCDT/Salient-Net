# train from scratch
if true; then
python -u ./main.py --lr 0.1 --ckpt-dir ./checkpoint/salientnet101-cifar100 --dataset cifar100 --batch-size 50 --backbone salientnet --layers-num 101 >./log/SalientNet101-cifar100.log
fi


# reume with special ckpt
if false; then
python -u ./main.py --lr 0.1 --ckpt-dir ./checkpoint/resnet101-cifar100 --pretrain-model ckpt --dataset cifar100 --batch-size 50 --backbone resnet --layers-num 101 >./log/ResNet101-cifar100-resume.log
fi