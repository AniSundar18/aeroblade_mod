
REAL_DIR='/nobackup/anirudh/edit_repo/aeroblade/data/raw/real/00000'
FAKE_DIR="/nobackup/anirudh/edit_repo/aeroblade/data/raw/generated/runwayml-stable-diffusion-v1-5-ViT-L-14-openai"





#CUDA_VISIBLE_DEVICES=3,4,5 python train_classifier.py --batch-size 20 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "lpips_vgg_2" --val_amount 360 --amount 800 --world-size 3 --out_dim 1 --save_dir '../checkpoints/attn_sample_lpips_vgg2_multiaug/' --real-dir=$REAL_DIR --fake-dirs=$FAKE_DIR --use_cnn --spatial --use_nested --in_dim 256


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python train_classifier.py --batch-size 1 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_8" --val_amount 360 --amount 800 --world-size 8 --out_dim 1 --save_dir '../checkpoints/attn_sample_lay8_multiaug_oneway/' --real-dir=$REAL_DIR --fake-dirs=$FAKE_DIR --use_attn --spatial --use_nested #--pre_rez 224 #--in_dim 2048 --use_cat


#CUDA_VISIBLE_DEVICES=2,3,5 python train.py --batch-size 50 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' --distance-metrics "CLIP:ViT-L/14_12" --val_amount 300 --use_mlp --amount 10000 --world-size 3 --save_dir '../checkpoints/mlp_jpg_redcaps/' --real-dir "/nobackup3/anirudh/datasets/redcaps/samples" --fake-dirs "/nobackup3/anirudh/datasets/laion_generated_images/1_fake"
