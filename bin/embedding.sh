CURRENT_DIR=$(pwd)
BIN_DIR=$(dirname "$BASH_SOURCE")
EXPERIMENTS_DIR="$BIN_DIR/../experiments"

cd $EXPERIMENTS_DIR
echo "Running Word Embeddings"
python doc_embeddings.py -d sale -n 30 -pc -e word

echo "Running Bert Embeddings"
python doc_embeddings.py -d sale -n 30 -pc -e bert

echo "Running Bert Embeddings Fine Tuning"
python doc_embeddings.py -d sale -n 30 -pc -ft -e bert
cd $CURRENT_DIR
