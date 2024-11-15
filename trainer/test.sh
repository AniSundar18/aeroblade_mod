REAL_FACES='/nobackup3/anirudh/datasets/whichfaceisreal/test_cnn/0_real_resized'
INV_FACES='/nobackup3/anirudh/datasets/whichfaceisreal/ddim'
cls_ckpt='ckpts/mlp_12_cls.pth'
attn_ckpt='ckpts/mlp_8_spatial_attn.pth'

CUDA_VISIBLE_DEVICES=1 python evaluate_cls.py --batch-size 30 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_8" --amount 600 --world-size 1 --checkpoint $attn_ckpt --real-dir=$REDCAPS --fake-dirs="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_webp" --use_attn --spatial

#CUDA_VISIBLE_DEVICES=1 python evaluate_cls.py --batch-size 40 --num-workers 0 --repo-ids 'stabilityai/stable-diffusion-2-1' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint $cls_ckpt --real-dir=$REDCAPS --fake-dirs="/nobackup3/anirudh/datasets/nights_subset/og_jpg" --use_mlp_cls --num_layers=2 #--vae_path "stabilityai/sd-vae-ft-mse"




