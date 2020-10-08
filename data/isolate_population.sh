#!/bin/bash
# Example Usage: ./isolate_population.sh chr20_kgp_abridged.vcf.gz kgp_samples.csv CEU
VCF_FILE=$1
POPULATION_FILE=$2
POPULATION=$3
FILE_NAME=$(basename $VCF_FILE)
OUTPUT_VCF_FILE="$POPULATION.${FILE_NAME}"
POPULATION_SAMPLES=""
SAMPLES=$(bcftools query -l $VCF_FILE)
while read p; do
    SAMPLE=$(echo "$p" | grep -oE "^[A-Za-z0-9\-]+,$POPULATION" | grep -oE "^[A-Za-z0-9\-]+")
    if ! [ -z "$SAMPLE" ] && [[ "$SAMPLES" == *"$SAMPLE"* ]]; then
        POPULATION_SAMPLES+="$SAMPLE,"
    fi
done < $POPULATION_FILE
POPULATION_SAMPLES=${POPULATION_SAMPLES%,}
bcftools view -O z -o $OUTPUT_VCF_FILE -s $POPULATION_SAMPLES $VCF_FILE
bcftools index -t $OUTPUT_VCF_FILE
