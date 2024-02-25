ref_count=500
wmkey_len=512

max_gen_len=256
data_path=com_diverse_data.json
prompt_type=completion
model=llama-2-7b
ppl_model=llama-2-13b
temperature=1
batch_size=100
attack_name=none
attack_param=0
repetition=1
methods_str="openaiNg#openaiNg_d5#openaiNg_d10#openaiNg_d15#openaiNg_d20#openaiNg_d25#openaiNg_d30#openaiNg_d35#openaiNg_d40#openaiNg_s10#openaiNg_s20#openaiNg_s30#openaiNg_s50#openaiNg_s70#openaiNg_s100#openaiNg_s150#openaiNg_s200#openaiNg_t0.1#openaiNg_t0.2#openaiNg_t0.3#openaiNg_t0.4#openaiNg_t0.5#gumbelsoftNg#gumbelsoftNg_d5#gumbelsoftNg_d10#gumbelsoftNg_d15#gumbelsoftNg_d20#gumbelsoftNg_d25#gumbelsoftNg_d30#gumbelsoftNg_d35#gumbelsoftNg_d40#gumbelsoftNg_s10#gumbelsoftNg_s20#gumbelsoftNg_s30#gumbelsoftNg_s50#gumbelsoftNg_s70#gumbelsoftNg_s100#gumbelsoftNg_s150#gumbelsoftNg_s200#gumbelsoftNg_t0.1#gumbelsoftNg_t0.2#gumbelsoftNg_t0.3#gumbelsoftNg_t0.4#gumbelsoftNg_t0.5"
table=com_diverse

drops="5 10 15 20 25 30 35 40"
shifts="10 20 30 50 70 100 150 200"
temperatures="0.1 0.2 0.3 0.4 0.5"

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
python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method gumbelsoftNg     --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature $temperature  --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table
python main_watermark.py --model_name $model --prompt_type $prompt_type --json_path data/${data_path}  --batch_size $batch_size --method unwatermarked    --max_gen_len ${max_gen_len} --seeding hash --seed 42 --ngram 3    --temperature $temperature --attack_name $attack_name --attack_param $attack_param --repetition $repetition --table_name $table

python evaluation/eval_diverse.py --table_name $table --methods $methods_str

python evaluation/collect_diversity.py --methods $methods_str
python evaluation/latex.py --diverse_path com_diverse.txt --detect_path table1_ana.txt --task Com
