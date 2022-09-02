CURRENT_DIR=$(pwd)
BIN_DIR=$(dirname "$BASH_SOURCE")
EXPERIMENTS_DIR="$BIN_DIR/../experiments"

cd $EXPERIMENTS_DIR
python bow.py -d sale -tf -n 30
cd $CURRENT_DIR
