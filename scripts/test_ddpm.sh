ulimit -n 2048
python test_ddpm.py sample-cond --n-steps 500 \
                                --device gpu:0,1,2,3 \
                                --save-path ~/ddpm_samples_celebahq_128_nsamples5k_form2_folder2/ \
                                --num-samples 5000 \
                                --compare False \
                                --seed 1 \
                                --batch-size 16 \
                                --n-workers 8 \
                                --use-concat True \
                                --checkpoints "" \
                                ~/checkpoints/celebahq128/celebahq128_ae/vae-epoch\=189-train_loss\=0.00.ckpt \
                                ~/checkpoints/celebahq128/ddpm_celebahq128_form2/checkpoints/ddpmv2-epoch\=800-loss\=0.0058.ckpt

# python test_ddpm.py generate-recons --n-steps 300 \
#                                 --device gpu:0,1,2,3 \
#                                 --save-path ~/cond_inference_form2/ \
#                                 --seed 0 \
#                                 --num-samples 8 \
#                                 --batch-size 2 \
#                                 --use-concat True \
#                                 --compare False \
#                                 --image-size 128 \
#                                 ~/vaedm/checkpoints/vae-epoch\=189-train_loss\=0.00.ckpt \
#                                 ~/ddpm_128_truncation_1.0/checkpoints/ddpmv2-epoch\=800-loss\=0.0058.ckpt \
#                                 ~/vaedm/reconstructions/
