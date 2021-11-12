python main/eval/ddpm/sample_cond.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.evaluation.sample_prefix='gpu_1' \
                        dataset.ddpm.evaluation.device=\'gpu:1\' \
                        dataset.ddpm.evaluation.seed=3 \
                        dataset.ddpm.model.attn_resolutions=\'16,8,\' \
                        dataset.ddpm.model.n_heads=4 \
                        dataset.ddpm.model.skip_scale=True \
                        dataset.ddpm.evaluation.chkpt_path=\'/data/kushagrap20/sota-checkpoints/ddpmv2-cifar10_form1_scale=[-1,1]_9thNov_sota-epoch=1210-loss=0.0295.ckpt\' \
                        dataset.ddpm.evaluation.type='form1' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=64 \
                        dataset.ddpm.evaluation.save_path=\'/data/kushagrap20/sota-samples/ddpm_cifar10_form1_temp=1.0\' \
                        dataset.ddpm.evaluation.n_samples=2500 \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=False \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/data/kushagrap20/checkpoints_old/cifar10/cifar10_ae/checkpoints/vae-epoch\=500-train_loss\=0.00.ckpt\'