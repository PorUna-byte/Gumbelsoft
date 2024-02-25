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
attack_name=none
attack_param=0
repetition=5
mainexp_methods_str="marylandNg#openaiNg#dipmarkNg#gumbelsoftNg_t0.3#ITSNg"
anaexp_methods_str="openaiNg#openaiNg_d5#openaiNg_d10#openaiNg_d15#openaiNg_d20#openaiNg_d25#openaiNg_d30#openaiNg_d35#openaiNg_d40#openaiNg_s10#openaiNg_s20#openaiNg_s30#openaiNg_s50#openaiNg_s70#openaiNg_s100#openaiNg_s150#openaiNg_s200#openaiNg_t0.1#openaiNg_t0.2#openaiNg_t0.3#openaiNg_t0.4#openaiNg_t0.5#gumbelsoftNg#gumbelsoftNg_d5#gumbelsoftNg_d10#gumbelsoftNg_d15#gumbelsoftNg_d20#gumbelsoftNg_d25#gumbelsoftNg_d30#gumbelsoftNg_d35#gumbelsoftNg_d40#gumbelsoftNg_s10#gumbelsoftNg_s20#gumbelsoftNg_s30#gumbelsoftNg_s50#gumbelsoftNg_s70#gumbelsoftNg_s100#gumbelsoftNg_s150#gumbelsoftNg_s200#gumbelsoftNg_t0.1#gumbelsoftNg_t0.2#gumbelsoftNg_t0.3#gumbelsoftNg_t0.4#gumbelsoftNg_t0.5"
table=table3

drops="5 10 15 20 25 30 35 40"
shifts="10 20 30 50 70 100 150 200"
temperatures="0.1 0.2 0.3 0.4 0.5"

# Analysis experiment
for drop in $drops
do
    python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method openaiNg  --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3  --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table --drop_prob $drop --method_suffix _d$drop
    python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method gumbelsoftNg  --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3  --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table --drop_prob $drop --method_suffix _d$drop
done

for shift in $shifts
do
    python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method openaiNg  --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3  --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table --shift_max $shift --method_suffix _s$shift
    python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method gumbelsoftNg  --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3  --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table --shift_max $shift --method_suffix _s$shift
done

for tem in $temperatures
do
    python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method openaiNg --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3  --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table --tau $tem  --method_suffix _t$tem
    python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method gumbelsoftNg --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3  --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table --tau $tem   --method_suffix _t$tem
done

python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method openaiNg         --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table
python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method gumbelsoftNg     --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table

python evaluation/filter_res.py --stage 1 --length $max_gen_len --ngram 3  --nsamples 1000 --methods $anaexp_methods_str --table_name $table --repetition $repetition
 
for drop in $drops
do
    python main_detect.py --tokenizer $model --method openaiNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --method_suffix _d$drop
    python main_detect.py --tokenizer $model --unwatermarked true --method openaiNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --method_suffix _d$drop 
    python main_detect.py --tokenizer $model --method gumbelsoftNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --method_suffix _d$drop
    python main_detect.py --tokenizer $model --unwatermarked true --method gumbelsoftNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --method_suffix _d$drop   
done

for shift in $shifts
do
    python main_detect.py --tokenizer $model --method openaiNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --shift_max $shift  --method_suffix _s$shift
    python main_detect.py --tokenizer $model --unwatermarked true --method openaiNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --shift_max $shift  --method_suffix _s$shift
    python main_detect.py --tokenizer $model --method gumbelsoftNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --shift_max $shift  --method_suffix _s$shift
    python main_detect.py --tokenizer $model --unwatermarked true --method gumbelsoftNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --shift_max $shift  --method_suffix _s$shift
done

for tem in $temperatures
do
    python main_detect.py --tokenizer $model --method openaiNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --method_suffix _t$tem
    python main_detect.py --tokenizer $model --unwatermarked true --method openaiNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --method_suffix _t$tem
    python main_detect.py --tokenizer $model --method gumbelsoftNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --method_suffix _t$tem
    python main_detect.py --tokenizer $model --unwatermarked true --method gumbelsoftNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --method_suffix _t$tem
done

python main_detect.py --tokenizer $model --method gumbelsoftNg --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table  
python main_detect.py --tokenizer $model --unwatermarked true --method gumbelsoftNg --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table 

python main_detect.py --tokenizer $model --method openaiNg      --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table
python main_detect.py --tokenizer $model --unwatermarked true --method openaiNg     --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table 

python evaluation/filter_res.py --stage 2 --length $max_gen_len --ngram 3  --nsamples 1000 --methods $anaexp_methods_str --table_name $table --repetition $repetition

python evaluation/eval_acc.py --methods $anaexp_methods_str --table_name $table --repetition $repetition
python evaluation/eval_ppl.py --ppl_model $ppl_model --methods $anaexp_methods_str --table_name $table --repetition $repetition
python evaluation/collect_results.py --table_name $table --methods $anaexp_methods_str --repetition $repetition


# Detectability experiment 
python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method marylandNg       --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table --gamma 0.1 --delta 2 
python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method openaiNg         --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table
python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method dipmarkNg        --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table --alpha 0.45 
python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method gumbelsoftNg     --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature 0          --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table
python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method ITSNg            --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table
python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method unwatermarked    --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table

python evaluation/filter_res.py --stage 1 --length $max_gen_len --ngram 3  --nsamples 1000 --methods $mainexp_methods_str --table_name $table --repetition $repetition

python main_detect.py --tokenizer $model --method marylandNg    --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --gamma 0.1 
python main_detect.py --tokenizer $model --method openaiNg      --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table
python main_detect.py --tokenizer $model --method dipmarkNg     --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --gamma 0.5    
python main_detect.py --tokenizer $model --method gumbelsoftNg  --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table               
python main_detect.py --tokenizer $model --method ITSNg         --ngram 3  --seeding hash --seed 42  --repetition $repetition --table_name $table --ref_count $ref_count --natural_text_path $natural_text_path --max_gen_len $max_gen_len --wmkey_len $wmkey_len 

python main_detect.py --tokenizer $model --unwatermarked true --method marylandNg   --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table --gamma 0.1
python main_detect.py --tokenizer $model --unwatermarked true --method openaiNg     --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table               
python main_detect.py --tokenizer $model --unwatermarked true --method dipmarkNg    --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table --gamma 0.5
python main_detect.py --tokenizer $model --unwatermarked true --method gumbelsoftNg --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table               
python main_detect.py --tokenizer $model --unwatermarked true --method ITSNg        --ngram 3  --seeding hash  --seed 42  --repetition $repetition --table_name $table --ref_count $ref_count --natural_text_path $natural_text_path --max_gen_len $max_gen_len --wmkey_len $wmkey_len

python evaluation/filter_res.py --stage 2 --length $max_gen_len --ngram 3  --nsamples 1000 --methods $mainexp_methods_str --table_name $table --repetition $repetition

python evaluation/eval_acc.py --methods $mainexp_methods_str --table_name $table --repetition $repetition
python evaluation/eval_ppl.py --ppl_model $ppl_model --methods $mainexp_methods_str --table_name $table --repetition $repetition
python evaluation/collect_results.py --table_name $table --methods $mainexp_methods_str --repetition $repetition