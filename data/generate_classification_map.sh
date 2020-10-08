#!/bin/bash
# Example Usage: ./generate_classification_map.sh kgp_samples.csv
DATA_FILE=$1
OUTPUT_FILE=classification_map.tsv
while read p; do
    echo "$p" | grep -oE "^[A-Za-z0-9\-]+,[A-Za-z0-9]+" | tr "," "\t" >> $OUTPUT_FILE
done < $DATA_FILE
sed -i '1d' $OUTPUT_FILE
