
REAL_DIR="REPLACE WITH PATH TO REAL IMAGES"
FAKE_DIR="REPLACE WITH PATH TO FAKE IMAGES"



# TRAINING DETAILS EXPLAINATION
# --distance-metrics "CLIP:ViT-L/14_8" tells indicates to extract feature representations from the 8th layer of the CLIP ViT
# --use_attn tells the code to use a attention head on top of the extracted features, applied along with --spatial
# --spatial makes the network to use all tokens, not using it would just use the CLS token
# --use_mlp would keep a simple MLP classifier on top of the extracted features, this is done when --spatial is not given
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python train_classifier.py --batch-size 1 --num-workers 0 --repo-ids 'runwayml/stable-diffusion-v1-5' 
                                                                  --distance-metrics "CLIP:ViT-L/14_8" --val_amount 360 --amount 800 --world-size 8 
                                                                  --out_dim 1 --save_dir '../checkpoints/attn_sample_lay8_multiaug_oneway/' 
                                                                  --real-dir=$REAL_DIR --fake-dirs=$FAKE_DIR --use_attn --spatial --use_nested #--pre_rez 224 #--in_dim 2048 --use_cat


