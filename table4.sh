#!/bin/bash
ref_count=500
wmkey_len=256

max_gen_len=128
data_path=alpaca_data.json
natural_text_path=data/c4_ref.json
prompt_type=chat
model=llama-2-7b-chat
ppl_model=llama-2-13b-chat
temperature=1
batch_size=100
attack_param=0.5
repetition=5
methods_str="marylandNg#openaiNg#dipmarkNg#ITSNg#gumbelsoftNg_t0.3"
table=table4

python evaluation/attack.py --methods $methods_str --attack_unwatermarked true --read_table_name table1 --write_table_name table2 --repetition $repetition --attack_param $attack_param

python main_detect.py --tokenizer $model --method marylandNg    --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --gamma 0.1 
python main_detect.py --tokenizer $model --method openaiNg      --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table
python main_detect.py --tokenizer $model --method dipmarkNg     --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --gamma 0.5    
python main_detect.py --tokenizer $model --method gumbelsoftNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --method_suffix "_t0.3"               
python main_detect.py --tokenizer $model --method ITSNg         --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --ref_count $ref_count --natural_text_path $natural_text_path --max_gen_len $max_gen_len --wmkey_len $wmkey_len 

python main_detect.py --tokenizer $model --unwatermarked true --method marylandNg   --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table --gamma 0.1
python main_detect.py --tokenizer $model --unwatermarked true --method openaiNg     --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table               
python main_detect.py --tokenizer $model --unwatermarked true --method dipmarkNg    --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table --gamma 0.5
python main_detect.py --tokenizer $model --unwatermarked true --method gumbelsoftNg --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table --method_suffix "_t0.3"           
python main_detect.py --tokenizer $model --unwatermarked true --method ITSNg        --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table --ref_count $ref_count --natural_text_path $natural_text_path --max_gen_len $max_gen_len --wmkey_len $wmkey_len

python evaluation/eval_acc.py --methods $methods_str --table_name $table --repetition $repetition
python evaluation/eval_ppl.py --ppl_model $ppl_model --methods $methods_str --table_name $table --repetition $repetition
python evaluation/collect_results.py --table_name $table --methods $methods_str --repetition $repetition

