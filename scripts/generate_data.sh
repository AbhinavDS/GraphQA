# Script that generates the data split which will be used for training and evaluating the network architecture. It also calls the modules that perform preprocessing on the dataset.
# Written specifically for GQA dataset and hence, not data agnostic

# Assumptions:
# The data directory already contains the merged object and image features.
# The script currently selects the subset of data from the balanced splits of the dataset that have been provided.

# The filtering logic has to coded in the src/utils/filter_data.py module for now. It is parameterized.

DATA_DIR=$1
OUT_DATA_DIR=$2
WORD2VEC_PATH=$3
PCT=$4

QA_DIR="${DATA_DIR}/qa/"
SG_DIR="${DATA_DIR}/sceneGraphs/"
FEATURES_DIR="${DATA_DIR}/features/"
SPLIT='balanced'

cd src/
python -m utils.filter_data --dataset $SPLIT --inp_qa_data_dir $QA_DIR --out_data_dir $OUT_DATA_DIR --pct $PCT --inp_sg_data_dir $SG_DIR

# Do the necessary preprocessing required by the network architecture
python -m utils.qa_preprocessing --input_questions_path "${OUT_DATA_DIR}/${SPLIT}_train_data.json" --output_vocab_path "${OUT_DATA_DIR}/qa_vocab.json" --meta_data_path "${OUT_DATA_DIR}/meta_data.json" --inp_word2vec_path $WORD2VEC_PATH --out_word2vec_path "${OUT_DATA_DIR}/glove.300d.json"

python -m utils.sg_preprocessing --input_relations_path "${OUT_DATA_DIR}/train_sceneGraphs.json" --output_vocab_path "${OUT_DATA_DIR}/sg_vocab.json" --meta_data_path "${OUT_DATA_DIR}/meta_data.json"