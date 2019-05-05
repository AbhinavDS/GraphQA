# Script that generates the data required for Learning the Scene Graph Model

DATA_DIR=$1
OUT_DIR="${DATA_DIR}/vg_data/"
IMG_DIR="/scratch/cluster/abhinav/gnlp/gqa_data/images/gqa/"

echo $DATA_DIR
cd src/
python -m utils.create_schema --data_dir $DATA_DIR --out_dir $OUT_DIR
python -m utils.convert_to_imdb --image_dir $IMG_DIR --imh5_dir $OUT_DIR --metadata_input "${OUT_DIR}/img_metadata.json"
python -m utils.convert_to_roidb --base_dir $OUT_DIR