
REAL_FACES='/nobackup3/anirudh/datasets/whichfaceisreal/test_cnn/0_real'
INV_FACES='/nobackup3/anirudh/datasets/whichfaceisreal/ddim'
REAL_PATH='/nobackup3/anirudh/aeroblade/data/raw/real/00000'
FAKE_PATH='/nobackup3/anirudh/aeroblade/data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai'
INV_REAL_PATH='/nobackup3/anirudh/aeroblade/data/inversions/real_800_inv'
DIST_DICT_DIR='/nobackup3/anirudh/aeroblade/data/distances' 
EXP_ID="rvi_vanilla"
SAVE_SPAT_DIST="${DIST_DICT_DIR}/${EXP_ID}"


#CUDA_VISIBLE_DEVICES=2 python experiments/04_study_resizing.py --real-dir=$FAKE_PATH --fake-dirs=$FAKE_PATH --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authorsfakes_bicub_0.8_rzrecon" --resize_ratio 0.8 --downsize_style 'bicubic' --upsize_style 'bicubic' --amount 800 --do_over --transforms 'resize'

#python experiments/04_study_resizing.py --real-dir=$REAL_PATH --fake-dirs=$REAL_PATH --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authorsreals_bicub_0.9_rzrecon" --resize_ratio 0.9 --downsize_style 'bicubic' --upsize_style 'bicubic' --amount 800 --do_over --transforms 'resize'

#python experiments/01_detect.py --real-dir=$INV_FACES_PATH --fake-dirs=$INV_REAL_PATH --batch-size 5 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id=$EXP_ID --amount 5 --do_over 


CUDA_VISIBLE_DEVICES=2 python experiments/01_detect.py --real-dir=$REAL_PATH --fake-dirs='/nobackup3/anirudh/datasets/laion_generated_images/1_fake_train' --batch-size 5 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_jedtrain_postopt" --amount 30 #--do_over --optimize  

#python experiments/04_study_resizing.py --real-dir=$REAL_PATH --fake-dirs=$FAKE_PATH --batch-size 5 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authorsreals_bicub_0.8" --resize_ratio 0.80 --downsize_style 'bicubic' --upsize_style 'bicubic' --amount 50 --do_over --transforms 'resize' --optimize

#python experiments/01_detect.py --real-dir '/nobackup3/anirudh/aeroblade/data/raw/real/00000' --fake-dirs '/nobackup3/anirudh/aeroblade/data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai' --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authors" --amount 800 --do_over

#python experiments/01_detect.py --real-dir '/nobackup3/anirudh/aeroblade/data/raw/real/00000' --fake-dirs '/nobackup3/anirudh/aeroblade/data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai' --batch-size 20 --repo-ids 'runwayml/stable-diffusion-v1-5' --experiment-id "runway_authors_bilin_rcrz_0.8_post" --amount 800 --do_over --transforms 'resize-recon_down-512' --scale 0.8 --post_transform
