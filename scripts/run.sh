python run_classification.py --model="meta-llama/Llama-3.1-8B-Instruct" --dataset="banking77" --all_shots="8" \
 --approx --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="rand"  --subsample_test_set=100 --approx  --bs=1 --gpu_id=1 --api_num_logprob=1000 \
 --calibration="TC" --tc_input_dim=13 --approx --calibrator_model_path="./calibration/models/meta-llama_Llama-3.1-8B-Instruct/snli_sst5_rte_agnews_trec/calibrator"

python -m calibration.generate_calibration_dataset --model="meta-llama/Llama-3.1-8B-Instruct" --dataset="sst5" --train_size=5000 --test_size=1000 --num_shots=20 --gpu_id=0 --api_num_logprob=1000
# python run_classification.py --model="google/gemma-3-12b-it" dataset="snli" --all_shots="8"  --approx --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="rand" --gpu_id=1

python -m calibration.train --llms="meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, openai/gpt-oss-20b" --datasets="snli, sst5, rte, agnews, trec, dbpedia_l2, toxic_chat, goemotions, dbpedia_l2, newsgroups" --shots_start=10 --iterations=10000 --batch_size=8 --eval_iter=800 --lr=0.00001 --temp_augment --label_augment

python -m calibration.eval --llms="meta-llama/Llama-3.1-8B-Instruct" --datasets="amazon_counterfactual, banking77, commonsense_qa, dbpedia_l1, massive_intent, metatool, qqp, when2call, wikitoxic, wildguard" --gpu_id=1
python -m calibration.eval --llms="gemini-2.5-flash" --datasets="amazon_counterfactual, qqp, wikitoxic" --gpu_id=1 --shots_start=10 --llm_agnostic  --model_name="calibrator_simple" --plot_results --plot_confidence_band

for d in */; do
  printf "%s %s\n" "$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)" "$d"
done | sort -nr | awk '{print strftime("%Y-%m-%d %H:%M:%S", $1), $2}'

find . -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n'

find . -type f -printf '%T@ %p\n' | sort -n | tail -20 | cut -d' ' -f2-
