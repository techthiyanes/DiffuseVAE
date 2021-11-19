# CelebAMaskHQ training
python main/train_postfit.py +dataset=cifar10/train \
                     dataset.post_vae.data.root='/data1/kushagrap20/cifar10_latents.npy' \
                     dataset.post_vae.model.hidden_dims='[512, 512, 512, 256, 256]' \
                     dataset.post_vae.model.latent_dim=512 \
                     dataset.post_vae.training.batch_size=256 \
                     dataset.post_vae.training.epochs=1000 \
                     dataset.post_vae.training.device=\'gpu:0,1\' \
                     dataset.post_vae.training.results_dir=\'/data1/kushagrap20/post_vae_cifar10_alpha=1.0/\' \
                     dataset.post_vae.training.workers=4 \
                     dataset.post_vae.training.chkpt_prefix='post_vae_alpha' \
