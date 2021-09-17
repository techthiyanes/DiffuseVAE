python test_ddpm.py interpolate-vae --n-steps 500 \
                                --device gpu:3 \
                                --save-path ~/cond-inference/ \
                                --seed 0 \
                                --reuse-epsilon False \
                                --n-interpolate 10 \
                                ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
                                ~/ddpm_128_truncation_1.0/checkpoints/ddpmv2-epoch\=983-loss\=0.0093.ckpt


# python test_ddpm.py interpolate-ddpm --n-steps 500 \
#                                 --device gpu:3 \
#                                 --save-path ~/cond-inference/ \
#                                 --seed 2 \
#                                 --reuse-epsilon False \
#                                 --n-interpolate 10 \
#                                 ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
#                                 ~/ddpm_128_truncation_1.0/checkpoints/ddpmv2-epoch\=983-loss\=0.0093.ckpt