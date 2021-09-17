# python test_ddpm.py sample-cond --n-steps 500 \
#                                 --device gpu:3 \
#                                 --save-path ~/cond-inference_without_concat/ \
#                                 --num-samples 1 \
#                                 --compare False \
#                                 --seed 0 \
#                                 ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
#                                 ~/ddpm_128_truncation_1.0_without_concat/checkpoints/ddpmv2-epoch\=127-loss\=0.0301.ckpt

python test_ddpm.py generate-recons --n-steps 500 \
                                --device gpu:3 \
                                --save-path ~/cond_inference/ \
                                --seed 0 \
                                --reuse-epsilon False \
                                --use-concat True \
                                --n-samples 8 \
                                ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
                                ~/ddpm_128_truncation_1.0/checkpoints/ddpmv2-epoch\=983-loss\=0.0093.ckpt \
                                ~/vaedm/reconstructions/
