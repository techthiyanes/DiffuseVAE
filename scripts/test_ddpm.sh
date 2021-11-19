# Formulation-1
# python main/eval/ddpm/sample_cond.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_2' \
#                         dataset.ddpm.evaluation.save_mode='numpy' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/ddpmv2-celebamaskhq_24thOct-epoch=259-loss=0.0054.ckpt\' \
#                         dataset.ddpm.evaluation.type='form1' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.device=\'gpu:2\' \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cmhq_form1_temp=1.0_5ksamples_seed0\' \
#                         dataset.ddpm.evaluation.n_samples=5000 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=False \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/vae-epoch=189-train_loss=0.00.ckpt\'

# Formulation-2
# python main/eval/ddpm/sample_cond.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_1' \
#                         dataset.ddpm.evaluation.save_mode='numpy' \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/ddpmv2-celebamaskhq_31stOct_form2_scale[-11]-epoch=253-loss=0.0127.ckpt\' \
#                         dataset.ddpm.evaluation.type='form2' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.device=\'gpu:1\' \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cmhq_form2_temp=1.0_5ksamples_seed0\' \
#                         dataset.ddpm.evaluation.n_samples=5000 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.save_vae=False \
#                         dataset.ddpm.evaluation.workers=1 \
#                         dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/vae-epoch=189-train_loss=0.00.ckpt\'

# Unconditional
# python main/eval/ddpm/sample.py +dataset=celebamaskhq128/test \
#                         dataset.ddpm.data.norm=True \
#                         dataset.ddpm.data.image_size=128 \
#                         dataset.ddpm.evaluation.sample_prefix='gpu_0' \
#                         dataset.ddpm.evaluation.save_mode='numpy' \
#                         dataset.ddpm.evaluation.seed=0 \
#                         dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cmhq/ddpmv2-celebamaskhq_1stNov_uncond_scale[-11]-epoch=268-loss=0.0021.ckpt\' \
#                         dataset.ddpm.evaluation.type='uncond' \
#                         dataset.ddpm.evaluation.temp=1.0 \
#                         dataset.ddpm.evaluation.batch_size=64 \
#                         dataset.ddpm.evaluation.device=\'gpu:0\' \
#                         dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cmhq_uncond_5ksamples_seed0\' \
#                         dataset.ddpm.evaluation.n_samples=5000 \
#                         dataset.ddpm.evaluation.n_steps=1000 \
#                         dataset.ddpm.evaluation.workers=1

python main/eval/ddpm/sample_cond.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.n_heads=1 \
                        dataset.ddpm.evaluation.variance='fixedsmall' \
                        dataset.ddpm.evaluation.seed=3 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_3' \
                        dataset.ddpm.evaluation.device=\'gpu:3\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/ddpmv2-cifar10_form1_scale=[-1,1]_15thNov_sota-epoch=1141-loss=0.0661.ckpt\' \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=128 \
                        dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cifar10_form1_temp_test_2_10k_fixedsmall\' \
                        dataset.ddpm.evaluation.n_samples=4000 \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=False \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\'