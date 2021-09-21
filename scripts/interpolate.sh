# python test_ddpm.py interpolate-vae --n-steps 1000 \
#                                 --device gpu:3 \
#                                 --save-path ~/cond-inference/ \
#                                 --seed 99 \
#                                 --reuse-epsilon False \
#                                 --n-interpolate 10 \
#                                 ~/checkpoints/celebahq128/celebahq128_ae/vae-epoch\=189-train_loss\=0.00.ckpt \
#                                 ~/checkpoints/celebahq128/ddpm_celebahq128_form2/checkpoints/ddpmv2-epoch\=800-loss\=0.0058.ckpt


python test_ddpm.py interpolate-ddpm --n-steps 1000 \
                                --device gpu:3 \
                                --save-path ~/cond-inference/ \
                                --seed 99 \
                                --reuse-epsilon False \
                                --n-interpolate 10 \
                                ~/checkpoints/celebahq128/celebahq128_ae/vae-epoch\=189-train_loss\=0.00.ckpt \
                                ~/checkpoints/celebahq128/ddpm_celebahq128_form2/checkpoints/ddpmv2-epoch\=800-loss\=0.0058.ckpt