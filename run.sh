
REAL_FACES='/nobackup3/anirudh/datasets/whichfaceisreal/test_cnn/0_real'
INV_FACES='/nobackup3/anirudh/datasets/whichfaceisreal/ddim'
REAL_INF='/nobackup3/anirudh/aeroblade/data/inference/real_val'
FAKE_INF='/nobackup3/anirudh/aeroblade/data/inference/gen_val'
REAL_PATH='/nobackup3/anirudh/aeroblade/data/raw/real/00000_jpg_80'
FAKE_PATH='/nobackup3/anirudh/aeroblade/data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai'
REAL_RECON_PATH='/nobackup3/anirudh/aeroblade/data/reconstructions/e6e92d74162428b036c7702ce2d59297'
FAKE_RECON_PATH='/nobackup3/anirudh/aeroblade/data/reconstructions/d6aa5a6ce07daefdda39364511db89e5'
INV_REAL_PATH='/nobackup3/anirudh/aeroblade/data/inversions/real_800_inv'
DIST_DICT_DIR='/nobackup3/anirudh/aeroblade/data/distances' 
IP2P_DIR='/nobackup3/anirudh/datasets/instructpix2pix_dataset'
NIGHTS_PATH='/nobackup3/anirudh/datasets/dreamsim/dataset/dataset/nights/distort'
JED_FAKES='/nobackup3/anirudh/datasets/laion_generated_images/1_fake'
JED_FAKE_JPG='/nobackup3/anirudh/datasets/laion_generated_images/1_fake_train'
GUSTAV_RANDOM="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake_random"
GUSTAV_JPG="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake_jpg"
JED_CIFAR_RANDOM="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_random"
CIFAR_RANDOM="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_random"
CIFAR_JPG="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_jpg"
CIFAR="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake"
EXP_ID="rvi_vanilla"
GUSTAV="/nobackup3/anirudh/datasets/SDv1-5/gustavosta/res-512/guidance-7-5/images/1_fake"
REDCAPS="/nobackup3/anirudh/datasets/redcaps/samples"
REDCAPS_RANDOM="/nobackup3/anirudh/datasets/redcaps/samples_random"
SAVE_SPAT_DIST="${DIST_DICT_DIR}/${EXP_ID}"

#CUDA_VISIBLE_DEVICES=3 python experiments/06_multi_recon.py --real-dir=$REAL_PATH --real-recon-dir="/nobackup3/anirudh/aeroblade/data/reconstructions/fc77b0cd0314873220d15a9a7428c66c" --fake-dirs=$CIFAR_RANDOM --fake-recon-dirs="/nobackup3/anirudh/aeroblade/data/reconstructions/4de0fca203da70194a3d715fa22c5762" --batch-size 60 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 120 --experiment-id "runway_authors" --iterations 50 

#CUDA_VISIBLE_DEVICES=2 python experiments/04_study_resizing.py --real-dir=$FAKE_PATH --fake-dirs=$FAKE_PATH --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authorsfakes_bicub_0.8_rzrecon" --resize_ratio 0.8 --downsize_style 'bicubic' --upsize_style 'bicubic' --amount 800 --do_over --transforms 'resize'

#python experiments/04_study_resizing.py --real-dir=$REAL_PATH --fake-dirs=$REAL_PATH --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authorsreals_bicub_0.9_rzrecon" --resize_ratio 0.9 --downsize_style 'bicubic' --upsize_style 'bicubic' --amount 800 --do_over --transforms 'resize'

CUDA_VISIBLE_DEVICES=2 python experiments/01_detect.py --real-dir="/nobackup3/anirudh/datasets/redcaps/samples" --fake-dirs="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_jittered" --batch-size 60 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 600 --experiment-id "clipall_mlp_runway_redcapsvcifarjit" --distance-metrics "CLIP:ViT-L/14_-1" #--checkpoint '/nobackup3/anirudh/aeroblade/trainer/checkpoints/mlp_jpg/mlp_12_epoch_best.pth' 
#--spatial

#CUDA_VISIBLE_DEVICES=2 python experiments/01_detect.py --real-dir="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake" --fake-dirs="/nobackup3/anirudh/datasets/SDv1-5/cifar-100/res-512/guidance-7-5/images/1_fake_denoise" --batch-size 60 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 600 --experiment-id "clipall_runway_cifar_v_denoise" --distance-metrics "CLIP:ViT-L/14_12" 

#CUDA_VISIBLE_DEVICES=5 python experiments/01_detect.py --real-dir=$REAL_PATH --fake-dirs=$CIFAR_RANDOM --batch-size 60 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 120 --experiment-id "runway_rvf_inf_multi" --iterations 50  #--distance-metrics "CLIP:ViT-L/14_-1" --spatial

#CUDA_VISIBLE_DEVICES=3 python experiments/01_detect.py --real-dir=$REDCAPS --fake-dirs=$GUSTAV --batch-size 60 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 600 --experiment-id "runway_redvgustav_clipall_tokenwise"  --distance-metrics "CLIP:ViT-L/14_-1" --spatial

#CUDA_VISIBLE_DEVICES=2 python experiments/01_detect.py --real-dir=$REDCAPS --fake-dirs=$INV_FACES --batch-size 60 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 600 --experiment-id "runway_redvddim_faces_clipall_tokenwise"  --distance-metrics "CLIP:ViT-L/14_-1" --spatial

#CUDA_VISIBLE_DEVICES=3 python experiments/01_detect.py --real-dir=$REDCAPS --fake-dirs=$CIFAR --batch-size 60 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 800 --experiment-id "runway_redvcifar"  #--distance-metrics "CLIP:ViT-L/14_-1" #--vae_path 'stabilityai/sd-vae-ft-mse' --spatial --do_over

#CUDA_VISIBLE_DEVICES=1 python experiments/01_detect.py --real-dir=$REAL_PATH --fake-dirs=$IP2P_DIR --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 800 --experiment-id "runway_rvnip2p_clipall"  --distance-metrics "CLIP:ViT-L/14_-1" --vae_path 'stabilityai/sd-vae-ft-mse' --do_over

#CUDA_VISIBLE_DEVICES=0 python experiments/01_detect.py --real-dir=$REAL_PATH --fake-dirs=$FAKE_PATH --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 800 --experiment-id "runway_authors_tokenwise_clipall_0.5_bicub" --distance-metrics "CLIP:ViT-L/14_-1" --spatial --scale 0.5  --transform 'resize-recon_down-512'

#python experiments/01_detect.py --real-dir=$REAL_PATH --fake-dirs=$FAKE_PATH --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 40 --experiment-id "runway_authors_iter50" --iterations 50

#python experiments/01_detect.py --real-dir=$REAL_PATH --fake-dirs=$FAKE_PATH --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --amount 800 --experiment-id "runway_authors_rz_0.5_bilinear" --scale 0.5  --transform 'resize-recon_down-512' --do_over
#--do_over 

#CUDA_VISIBLE_DEVICES=2 python experiments/01_detect.py --real-dir=$REAL_PATH --fake-dirs=$NIGHTS_PATH --batch-size 13 --repo-ids 'stabilityai/stable-diffusion-2-1' --experiment-id "runway_real_nights_preopt_2.1_rz_1.25_bicub" --amount 800 --scale 1.25  --transform 'resize-recon_down-512'

#CUDA_VISIBLE_DEVICES=2 python experiments/01_detect.py --real-dir=$REAL_PATH --fake-dirs=$NIGHTS_PATH --batch-size 13 --repo-ids 'stabilityai/stable-diffusion-2-1' --experiment-id "runway_real_nights_preopt_2.1_rz_1.5_bicub" --amount 800 --scale 1.5  --transform 'resize-recon_down-512'

#CUDA_VISIBLE_DEVICES=2 python experiments/01_detect.py --real-dir=$IP2P_DIR --fake-dirs=$IP2P_DIR --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_ip2p"  --amount 5998 --vae_path 'stabilityai/sd-vae-ft-mse' #--transforms 'resize-recon_down-512' --scale 1.25 #--vae_path 'stabilityai/sd-vae-ft-mse'  #--do_over --optimize  




#python experiments/04_study_resizing.py --real-dir=$REAL_PATH --fake-dirs=$FAKE_PATH --batch-size 5 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authorsreals_bicub_0.8" --resize_ratio 0.80 --downsize_style 'bicubic' --upsize_style 'bicubic' --amount 50 --do_over --transforms 'resize' --optimize

#python experiments/01_detect.py --real-dir '/nobackup3/anirudh/aeroblade/data/raw/real/00000' --fake-dirs '/nobackup3/anirudh/aeroblade/data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai' --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authors" --amount 800 --do_over

#python experiments/01_detect.py --real-dir '/nobackup3/anirudh/aeroblade/data/raw/real/00000' --fake-dirs '/nobackup3/anirudh/aeroblade/data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai' --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authors_bilin_rcrz_0.8_post" --amount 800 --do_over --transforms 'resize-recon_down-512' --scale 0.8 --post_transform
