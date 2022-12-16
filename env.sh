export DS_DIR='/Users/hainguyen/Documents/AI4Code_data'

export RAW_DATA_DIR="$DS_DIR/ai4code"
export PROCESSED_DATA_DIR="$DS_DIR/proc-ai4code"
export SEED=42
export PICKLE_PROTOCOL=4

clearenv () {
    unset RAW_DATA_DIR
    unset PROCESSED_DATA_DIR
    unset SEED
    unset PICKLE_PROTOCOL
}