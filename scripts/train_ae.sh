# CIFAR10 settings
python train_ae.py --enc-block-config "32x7,32d2,32t16,16x4,16d2,16t8,8x4,8d2,8t4,4x3,4d4,4t1,1x3" \
                   --enc-channel-config "32:64,16:128,8:256,4:256,1:512" \
                   --dec-block-config "1x1,1u4,1t4,4x2,4u2,4t8,8x3,8u2,8t16,16x7,16u2,16t32,32x15" \
                   --dec-channel-config "32:64,16:128,8:256,4:256,1:512" \
                   --batch-size 32 \
                   --epochs 1000 \
                   --image-size 32 \
                   --workers 8 \
                   --device gpu:0,1,2,3 \
                   --dataset cifar10 \
                   --results-dir ~/cifar10_ae/ \
                   --seed 0 \
                   --restore-path ~/cifar10_ae/checkpoints/vae-epoch\=204-train_loss\=0.00.ckpt \
                   ~/datasets/