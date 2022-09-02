CURRENT_DIR=$(pwd)
BIN_DIR=$(dirname "$BASH_SOURCE")
EXPERIMENTS_DIR="$BIN_DIR/../../experiments"

cd $EXPERIMENTS_DIR
python bow-homes.py -b -n 30
cd $CURRENT_DIR
