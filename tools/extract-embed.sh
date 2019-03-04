#!/bin/bash

# Usage: ./extract-embed.sh [PARAM_FILE] [VOCAB_FILE] [source|target]

# Inputs
PARAM_FILE=${1}
VOCAB_JSON=${2}
SIDE=${3}

# Outputs
EMBED_TXT=${PARAM_FILE}.${SIDE}_embed_weight
VOCAB_TXT=${VOCAB_JSON::-5}.txt
OUTPUT_FILE=${EMBED_TXT}.vec

# Extract embed
echo "Extracting ${SIDE} embed..."
python -m sockeye.extract_parameters --names ${SIDE}_embed_weight --text-output --output ${PARAM_FILE} ${PARAM_FILE}

# Convert vocab file from json to txt
# 1. Counts dropped
# 2. Backslash removed from double quote ("), backslash (\)
# WARNING: Unicode combining class characters are not properly printed in the screen (look at the end of the previous line!)
echo "Converting vocab..."
cat ${VOCAB_JSON} | sed '1d;$d;s/\": .\+$//;s/\"//;s/ //g' | sed 's/^\\//' > ${VOCAB_TXT}

# Paste vocab and embed
echo "Writing .vec format..."
paste -d' ' ${VOCAB_TXT} ${EMBED_TXT} > ${OUTPUT_FILE}
sed -i "1s/^/$(wc -l ${VOCAB_TXT} | cut -d' ' -f1) $(head -n1 ${EMBED_TXT} | wc -w)\n/" ${OUTPUT_FILE}

# Finish
echo "Cleaning up..."
source deactivate
rm ${EMBED_TXT}
