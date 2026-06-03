#!/usr/bin/env bash
set -e

# Ollama configuration
export OLLAMA_MODEL=${OLLAMA_MODEL:-"qwen3:8b"}
export OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-"http://localhost:11434"}

debug=false

dataset=$1
config_file=$2

config_filename=$(basename -- "${config_file}")
config_filename="${config_filename%.*}"

debug_batch_size=1
batch_size=8
temperature=0

output=output/${dataset}/${OLLAMA_MODEL}/${config_filename}.jsonl
echo 'output to:' $output

prompt_type=""
engine=faiss
embedding_model="all-MiniLM-L6-v2"

if [[ ${dataset} == '2wikihop' ]]; then
    input="--input data/2wikimultihopqa"
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=500
    max_generation_len=256
elif [[ ${dataset} == 'strategyqa' ]]; then
    input="--input data/strategyqa/dev_beir"
    index_name=wikipedia_dpr
    fewshot=6
    max_num_examples=229
    max_generation_len=256
elif [[ ${dataset} == 'asqa' ]]; then
    prompt_type="--prompt_type general_hint_in_output"
    input="--input data/asqa/ASQA.json"
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=500
    max_generation_len=256
elif [[ ${dataset} == 'asqa_hint' ]]; then
    prompt_type="--prompt_type general_hint_in_input"
    dataset=asqa
    input="--input data/asqa/ASQA.json"
    index_name=wikipedia_dpr
    fewshot=8
    max_num_examples=500
    max_generation_len=256
elif [[ ${dataset} == 'wikiasp' ]]; then
    input="--input data/wikiasp"
    index_name=wikiasp
    fewshot=4
    max_num_examples=500
    max_generation_len=512
else
    echo "Unknown dataset: ${dataset}"
    exit 1
fi

# query api
if [[ ${debug} == "true" ]]; then
    python -m src.openai_api \
        --dataset ${dataset} ${input} ${prompt_type} \
        --config_file ${config_file} \
        --fewshot ${fewshot} \
        --search_engine ${engine} \
        --embedding_model ${embedding_model} \
        --index_name ${index_name} \
        --max_num_examples 100 \
        --max_generation_len ${max_generation_len} \
        --batch_size ${debug_batch_size} \
        --output test.jsonl \
        --num_shards 1 \
        --shard_id 0 \
        --debug
    exit
fi

python -m src.openai_api \
    --dataset ${dataset} ${input} ${prompt_type} \
    --config_file ${config_file} \
    --fewshot ${fewshot} \
    --search_engine ${engine} \
    --embedding_model ${embedding_model} \
    --index_name ${index_name} \
    --max_num_examples ${max_num_examples} \
    --max_generation_len ${max_generation_len} \
    --temperature ${temperature} \
    --batch_size ${batch_size} \
    --output ${output} \
    --num_shards 1 \
    --shard_id 0
