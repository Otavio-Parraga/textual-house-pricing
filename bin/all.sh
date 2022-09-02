BIN_DIR=$(dirname "$BASH_SOURCE")
LOG_DIR="$BIN_DIR/../logs"

echo "Running baseline.sh..."
sh $BIN_DIR/baseline.sh > $LOG_DIR/baseline.txt

echo "Running binary.sh..."
sh $BIN_DIR/binary.sh > $LOG_DIR/binary.txt

echo "Running count.sh..."
sh $BIN_DIR/count.sh > $LOG_DIR/count.txt

echo "Running tfidf-binary.sh..."
sh $BIN_DIR/tfidf-binary.sh > $LOG_DIR/tfidf-binary.txt

echo "Running tfidf.sh..."
sh $BIN_DIR/tfidf.sh > $LOG_DIR/tfidf.txt

echo "Running embeddings.sh"
sh $BIN_DIR/embeddings.sh > $LOG_DIR/embeddings.txt

echo "Running homes/baseline.sh..."
sh $BIN_DIR/homes/baseline.sh > $LOG_DIR/homes/baseline.txt

echo "Running homes/binary.sh..."
sh $BIN_DIR/homes/binary.sh > $LOG_DIR/homes/binary.txt

echo "Running homes/count.sh..."
sh $BIN_DIR/homes/count.sh > $LOG_DIR/homes/count.txt

echo "Running homes/tfidf-binary.sh..."
sh $BIN_DIR/homes/tfidf-binary.sh > $LOG_DIR/homes/tfidf-binary.txt

echo "Running homes/tfidf.sh..."
sh $BIN_DIR/homes/tfidf.sh > $LOG_DIR/homes/tfidf.txt

echo "Running homes/embeddings.sh"
sh $BIN_DIR/homes/embeddings.sh > $LOG_DIR/homes/embeddings.txt
