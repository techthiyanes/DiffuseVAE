# python third_party/fid_pytorch/fid.py --device cuda:2 --mode1 img --mode2 np ~/datasets/CelebAMask-HQ-128/ ~/ddpm_cmhq_form1_temp=1.0_5ksamples_seed0/1000/images/
# /data/kushagrap20/.local/share/virtualenvs/VAEDM-GADu0QCg/lib/python3.6/site-packages/cleanfid/stats/cifar10_legacy_pytorch_train_32.npz
python third_party/fid_pytorch/fid.py --device cuda:1 --mode1 np --mode2 img /data1/kushagrap20/cifar10_legacy_pytorch_train_32.npz /data1/kushagrap20/ddpm_cifar10_form1_temp_test_2_50k/1000/images/

# python eval/fid.py compute-fid-from-samples --num-batches 500 \
#                                             /data/kushagrap20/.local/share/virtualenvs/VAEDM-GADu0QCg/lib/python3.6/site-packages/cleanfid/stats/cifar10_legacy_tensorflow_train_32.npz \
#                                             /data/kushagrap20/ddpm_samples_cifar10_nsamples50k_uncond_fixedlarge_test2/1000/images