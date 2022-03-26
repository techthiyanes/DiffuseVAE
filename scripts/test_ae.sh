# python main/test.py reconstruct --device gpu:0 \
#                                 --dataset celeba \
#                                 --image-size 64 \
#                                 --save-path ~/vae_celeba64_recons/ \
#                                 --write-mode numpy \
#                                 ~/vae_celeba64_alpha\=1.0/checkpoints/vae-celeba64_alpha\=1.0-epoch\=245-train_loss\=0.0000.ckpt \
#                                 ~/datasets/img_align_celeba/


python main/test.py reconstruct --device gpu:0 \
                                --dataset celebahq \
                                --image-size 256 \
                                --save-path ~/test_celebahq_recons/ \
                                --write-mode image \
                                --num-samples 16 \
                                ~/vae_celebahq256_alpha\=1.0_Jan31/checkpoints/vae-celebahq256_alpha\=1.0_Jan31-epoch\=458-train_loss\=0.0000.ckpt \
                                ~/datasets/celeba_hq/

# python main/test.py reconstruct --device gpu:0 \
#                                 --dataset afhq \
#                                 --image-size 128 \
#                                 --save-path ~/reconstructions/afhq_reconsv2/ \
#                                 --write-mode numpy \
#                                 ~/vae_afhq_alpha\=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt \
#                                 ~/datasets/afhq/

# python main/test.py sample --device gpu:0 \
#                             --image-size 128 \
#                             --seed 0 \
#                             --num-samples 64 \
#                             --save-path ~/afhq_vae_samples1/ \
#                             --write-mode image \
#                             1024 \
#                             ~/vae_afhq_alpha\=1.0/checkpoints/vae-afhq_alpha=1.0-epoch=1499-train_loss=0.0000.ckpt \

# python main/test.py reconstruct --device gpu:0 \
#                            --num-samples 16 \
#                            --save-path ~/vae_alpha_1_0_samples/ \
#                            ~/checkpoints_old/celebahq128/celebahq128_ae/vae-epoch\=189-train_loss\=0.00.ckpt \
#                            ~/datasets/CelebAMask-HQ/