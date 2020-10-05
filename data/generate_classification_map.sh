#!/bin/bash
# kgp_samples.csv or similar file should be first argument
OUTPUT_FILE=classification_map.tsv
while read p; do
    echo "$p" | grep -oE "^[A-Za-z0-9]+,[A-Za-z0-9]+" | tr "," "\t" >> $OUTPUT_FILE
done < "$1"
sed -i '1d' "$1"
