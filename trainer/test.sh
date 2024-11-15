REAL_FACES='/nobackup3/anirudh/datasets/whichfaceisreal/test_cnn/0_real_resized'
INV_FACES='/nobackup3/anirudh/datasets/whichfaceisreal/ddim'
INV_FACES_WEBP='/nobackup3/anirudh/datasets/whichfaceisreal/ddim_webp'
JED_FAKE_JPG='/nobackup3/anirudh/datasets/laion_generated_images/1_fake_train'
REDCAPS="/nobackup3/anirudh/datasets/redcaps/samples"
REDCAPS_RANDOM="/nobackup3/anirudh/datasets/redcaps/samples_random"
JED_FAKE='/nobackup3/anirudh/datasets/laion_generated_images/1_fake'
REAL_INF='/nobackup3/anirudh/aeroblade/data/inference/real_val'
FAKE_INF='/nobackup3/anirudh/aeroblade/data/inference/gen_val'
REAL_DIR='/nobackup3/anirudh/aeroblade/data/raw/real/00000'
FAKE_DIR="/nobackup3/anirudh/aeroblade/data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai_jpg60"
FAKE_IP2P="/nobackup3/anirudh/datasets/ip2p_subset"
FAKE_IP2P_RANDOM="/nobackup3/anirudh/datasets/ip2p_subset_random"
NIGHTS="/nobackup3/anirudh/datasets/dreamsim/dataset/dataset/nights/distort"
GUSTAV_BIG="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-1024/guidance-7-5/images/1_fake"
CIFAR_RANDOM="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_random"
CIFAR_TESTRAND="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_testaug"
CIFAR_JPG="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_jpg"
CIFAR_WEBP="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_webp"
CIFAR_RZ="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_lanczos"
CIFAR="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake"
CIFAR_JITTER="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_jittered"
GUSTAV_JPG="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake_jpg"
GUSTAV="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake"
GUSTAV_RANDOM="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake_random"
GUSTAV_TESTRAND="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake_testaug"
GUSTAV_WEBP="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake_webp"
LAION_CAP_JPG="/nobackup3/anirudh/datasets/SDv1-5/laion_captions/res-512/guidance-7-5/images/1_fake_jpg"
#CUDA_VISIBLE_DEVICES=1 python tsne.py --batch-size 25 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_10" --amount 600 --world-size 1 --checkpoint 'sample.png' --real-dir=$REAL_DIR --fake-dirs=$FAKE_DIR 

#CUDA_VISIBLE_DEVICES=3 python evaluate_cls.py --batch-size 30 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_8" --amount 600 --world-size 1 --checkpoint '../checkpoints/attn_sample_lay8_multiaug/mlp_8_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-1024/guidance-7-5/images/1_fake" --use_attn --spatial --pre_rez 256

#CUDA_VISIBLE_DEVICES=3 python evaluate_cls.py --batch-size 60 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint '../checkpoints/mlp_sample_dropout_0.5_l2/mlp_12_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs=$CIFAR_JITTER --use_mlp_cls --num_layers=2

#CUDA_VISIBLE_DEVICES=3 python evaluate_dino.py --batch-size 30 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint '../checkpoints/mlp_sample_lay12_l2_dino/mlp_12_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs=$GUSTAV --use_mlp_cls --num_layers=2


#CUDA_VISIBLE_DEVICES=0 python evaluate_cls.py --batch-size 60 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint '../checkpoints/mlp_sample_lay12_multiaug_cat/mlp_12_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs="/nobackup3/anirudh/datasets/ddim_512" --use_mlp_cls --num_layers=2 --use_cat --in_dim 2048 #--vae_path "stabilityai/sd-vae-ft-mse"


#CUDA_VISIBLE_DEVICES=1 python evaluate_cls.py --batch-size 60 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint '../checkpoints/mlp_sample_lay12_l2_multiaug/mlp_12_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs="/nobackup3/anirudh/datasets/ddim_512" --use_mlp_cls --num_layers=2

#CUDA_VISIBLE_DEVICES=6 python evaluate_cls.py --batch-size 20 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_8" --amount 600 --world-size 1 --checkpoint '../checkpoints/attn_sample_lay8_multiaug_oneway/mlp_8_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs=$FAKE_INF --use_attn --spatial 

#--pre_rez 256 #--vae_path "stabilityai/sd-vae-ft-mse" #--use_mlp_cls --num_layers=2 #--vae_path "stabilityai/sd-vae-ft-mse"

#CUDA_VISIBLE_DEVICES=4 python evaluate_cls.py --batch-size 60 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint '../checkpoints/mlp_sample_lay12_l2_multiaug/mlp_12_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs=$INV_FACES_WEBP --use_mlp_cls --num_layers=2 #--use_cat --in_dim=2048 #--use_attn --spatial

CUDA_VISIBLE_DEVICES=1 python evaluate_cls.py --batch-size 30 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_8" --amount 600 --world-size 1 --checkpoint '../checkpoints/attn_sample_lay8_multiaug_rz/mlp_8_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_webp" --use_attn --spatial

#CUDA_VISIBLE_DEVICES=1 python evaluate_cls.py --batch-size 40 --num-workers 0 --repo-ids 'stabilityai/stable-diffusion-2-1' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint '../checkpoints/mlp_sample_lay12_l2_multiaug/mlp_12_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs="/nobackup3/anirudh/datasets/nights_subset/og_jpg" --use_mlp_cls --num_layers=2 #--vae_path "stabilityai/sd-vae-ft-mse"


#CUDA_VISIBLE_DEVICES=7 python evaluate_cls.py --batch-size 60 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_8" --amount 600 --world-size 1 --checkpoint '../checkpoints/attn_sample_lay8_multiaug/mlp_8_epoch_best.pth' --real-dir="/nobackup3/anirudh/datasets/lsun/0_real" --fake-dirs="/nobackup3/anirudh/datasets/lsun/0_real" --use_attn --spatial #--vae_path "stabilityai/sd-vae-ft-mse" #--use_mlp_cls --num_layers=2 #--vae_path "stabilityai/sd-vae-ft-mse"

#CUDA_VISIBLE_DEVICES=2 python evaluate_cls.py --batch-size 40 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 160 --world-size 1 --checkpoint '/nobackup3/anirudh/aeroblade/trainer/checkpoints/mlp_sample_dropout_0.5_l2/mlp_12_epoch_best.pth' --real-dir="/nobackup3/anirudh/datasets/domains/datasets/silhouette" --fake-dirs="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake" --use_mlp_cls --num_layers=2 #--use_distance

#CUDA_VISIBLE_DEVICES=2 python evaluate.py --batch-size 60 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --real-dir=$REDCAPS --fake-dirs=$GUSTAV_WEBP #--num_layers=2


#CUDA_VISIBLE_DEVICES=0 python evaluate_cls.py --batch-size 15 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint '/nobackup3/anirudh/aeroblade/trainer/checkpoints/mlp_sample_lay12_l2_onlyjpg/mlp_12_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs=$CIFAR_TESTRAND --num_layers=2 --use_mlp_cls


#CUDA_VISIBLE_DEVICES=2 python evaluate_cls.py --batch-size 60 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint '../checkpoints/mlp_sample_dropout_0.5_l2/mlp_12_epoch_best.pth' --real-dir=$REDCAPS --fake-dirs="/nobackup3/anirudh/datasets/SDv1-5/raise/res-512/guidance-7-5/images/1_fake_jpg" --num_layers=2 --use_mlp_cls


#CUDA_VISIBLE_DEVICES=2 python evaluate.py --batch-size 60 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --amount 600 --world-size 1 --checkpoint '../checkpoints/mlp/mlp_12_epoch_best.pth' --real-dir=$REAL_DIR --fake-dirs=$GUSTAV_JPG --use_mlp  
