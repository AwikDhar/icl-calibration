python run_classification.py --model="gemini-2.5-flash" --dataset="commonsense_qa" --all_shots="5"  --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="random"  --subsample_test_set=200 --calibration_set_size=100 --api_num_logprob=800 --gpu_ids=2 --calibration="TF, ICT, FS_ICT" --calibrator_name="calibrator_embed"

python -m calibration.generate_calibration_dataset --model="meta-llama/Llama-3.1-8B-Instruct" --dataset="sst5" --train_size=5000 --test_size=1000 --num_shots=20 --gpu_ids=0 --api_num_logprob=1000
# python run_classification.py --model="google/gemma-3-12b-it" dataset="snli" --all_shots="8"  --approx --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="rand" --gpu_id=1

python -m calibration.train --llms="meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, openai/gpt-oss-20b" --datasets="snli, sst5, rte, agnews, trec, toxic_chat, goemotions, dbpedia_l2, newsgroups, yahoo_answers" --shots_start=6 --iterations=20000 --batch_size=16 --eval_iter=800 --lr=0.00001 --gpu_id=0 --model_name="calibrator_embed2" --unseen_datasets="qqp, dbpedia_l1, banking77, yelp_reviews" --ablation_method="main" --num_seeds=10

python -m calibration.eval --llms="meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, openai/gpt-oss-20b" --datasets="amazon_counterfactual, banking77, commonsense_qa, dbpedia_l1, massive_intent, metatool, qqp, when2call, wikitoxic, wildguard" --gpu_id=1
python -m calibration.eval --llms="meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, openai/gpt-oss-20b" --datasets="banking77, dbpedia_l1, qqp, yelp_reviews" --gpu_id=0 --ablation_method="main" --num_seeds=10 --shots_start=5 --shots_end=15
python -m calibration.eval --llms="gemini-2.5-flash" --datasets="amazon_counterfactual, qqp, wikitoxic" --gpu_id=1 --shots_start=10 --llm_agnostic  --model_name="calibrator_simple" --plot_results --plot_confidence_band
python -m calibration.eval --llms="microsoft/Phi-4-mini-instruct" --datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --gpu_id=0 

CUDA_VISIBLE_DEVICES="1,2" python benchmark_moe.py  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --tp-size 2  --dtype fp8_w8a8 --tune  --batch-size 1 --save-dir "$CFG_DIR" --trust-remote-code

for d in */; do
  printf "%s %s\n" "$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)" "$d"
done | sort -nr | awk '{print strftime("%Y-%m-%d %H:%M:%S", $1), $2}'

find . -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n'

find . -type f -printf '%T@ %p\n' | sort -n | tail -20 | cut -d' ' -f2-

srun --gpus 4g.90gb:1 --cpus-per-task=32 --mem=128G –-time 12:00:00 --pty bash

python -m plot_results.tabulate_comparisons --models="microsoft/Phi-4-mini-instruct" --datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --num_seeds=10 --all_shots="5, 6, 8, 10, 12" --sampling_strategies="random" --calibration_methods="ICC, PERMUT_AVG, ICT, FS_ICT, TF" --metrics="ece, brier"

python -m plot_results.plot_comparisons --models="microsoft/Phi-4-mini-instruct" --datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --num_seeds=10 --all_shots="5, 6, 8, 10, 12" --sampling_strategies="random" --calibration_methods="ICC, PERMUT_AVG, ICT, FS_ICT, TF" --metrics="accuracy, ece, mce, brier"