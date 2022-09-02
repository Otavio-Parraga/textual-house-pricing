CURRENT_DIR=$(pwd)
BIN_DIR=$(dirname "$BASH_SOURCE")
BASELINE_DIR="$BIN_DIR/../baseline"

cd $BASELINE_DIR
python baseline.py -d sale
cd $CURRENT_DIR
