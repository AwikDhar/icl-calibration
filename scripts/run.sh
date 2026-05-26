python run_classification.py --model="microsoft/Phi-4-mini-instruct" --dataset="commonsense_qa" --all_shots="5, 6, 8, 10, 12"  --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="random"  --subsample_test_set=300 --api_num_logprob=800 --gpu_ids=0 --calibration="TF, ICT, FS_ICT" --calibrator_name="calibrator_embed"

python -m calibration.generate_calibration_dataset --model="meta-llama/Llama-3.1-8B-Instruct" --dataset="sst5" --train_size=5000 --test_size=1000 --num_shots=20 --gpu_ids=0 --api_num_logprob=1000
# python run_classification.py --model="google/gemma-3-12b-it" dataset="snli" --all_shots="8"  --approx --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="rand" --gpu_id=1

python -m calibration.train --llms="meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, openai/gpt-oss-20b" --datasets="snli, sst5, rte, agnews, trec, toxic_chat, goemotions, dbpedia_l2, newsgroups, yahoo_answers" --shots_start=6 --iterations=30000 --batch_size=16 --eval_iter=800 --lr=0.00001 --gpu_id=0 --model_name="calibrator_embed" --unseen_datasets="qqp, dbpedia_l1, banking77, yelp_reviews" --ablation_method="main" --num_seeds=10 --temp_augment

python -m calibration.eval --llms="meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, openai/gpt-oss-20b" --datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --gpu_id=0
python -m calibration.eval --llms="meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, openai/gpt-oss-20b" --datasets="banking77, dbpedia_l1, qqp, yelp_reviews" --gpu_id=0 --ablation_method="main" --num_seeds=10 --shots_start=5 --shots_end=15
python -m calibration.eval --llms="microsoft/Phi-4-mini-instruct" --datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --shots_start=5 --shots_end=15 --gpu_id=0 --model_name="calibrator_embed" --ablation_method="main" --num_seeds=20 --llm_agnostic
python -m calibration.eval --llms="microsoft/Phi-4-mini-instruct" --datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --shots_start=5 --shots_end=15  --model_name="calibrator_hd_ne" --ablation_method="half_data_no_embed" --num_seeds=10 --llm_agnostic
python -m calibration.eval --llms="gemini-2.5-flash" --datasets="amazon_counterfactual, qqp, wikitoxic" --gpu_id=1 --shots_start=10 --llm_agnostic  --model_name="calibrator_simple" --plot_results --plot_confidence_band

CUDA_VISIBLE_DEVICES="1,2" python benchmark_moe.py  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --tp-size 2  --dtype fp8_w8a8 --tune  --batch-size 1 --save-dir "$CFG_DIR" --trust-remote-code
CUDA_VISIBLE_DEVICES=0 python benchmark_moe.py --model="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8" --batch-size=1 --tp-size=1 --trust-remote-code --save-dir="/scratch/awikdhar/.conda/envs/calibvenv_boa/lib/python3.11/site-packages/vllm/model_executor/layers/fused_moe/configs/" --tune --dtype="fp8_w8a8"

for d in */; do
  printf "%s %s\n" "$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)" "$d"
done | sort -nr | awk '{print strftime("%Y-%m-%d %H:%M:%S", $1), $2}'

find . -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n'

find . -type f -printf '%T@ %p\n' | sort -n | tail -20 | cut -d' ' -f2-

srun --partition=Regular --gpus 4g.90gb:1 --cpus-per-task=20 --mem=96G –-time 1-00:00:00 --pty bash

python -m plot_results.tabulate_comparisons --models="microsoft/Phi-4-mini-instruct" --datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --num_seeds=10 --all_shots="5, 6, 8, 10, 12" --sampling_strategies="random" --calibration_methods="ICC, PERMUT_AVG, ICT, FS_ICT, TF" --metrics="ece, brier"
python -m plot_results.plot_comparisons --models="microsoft/Phi-4-mini-instruct" --datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --num_seeds=10 --all_shots="5, 6, 8, 10, 12" --sampling_strategies="random" --calibration_methods="ICC, PERMUT_AVG, ICT, FS_ICT, TF" --metrics="accuracy, ece, brier"
python -m plot_results.shotwise_comparisons --models="microsoft/Phi-4-mini-instruct" --datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --num_seeds=10 --all_shots="5, 6, 8, 10, 12" --sampling_strategies="random" --calibration_methods="ICC, PERMUT_AVG, ICT, FS_ICT, TF" --metrics="accuracy, brier, ece"
python -m plot_results.plot_baseline --models="microsoft/Phi-4-mini-instruct" --datasets="amazon_counterfactual, massive_intent, metatool, wikitoxic" --num_seeds=10 --all_shots="5, 6, 8, 10, 12" --sampling_strategies="random" --metric="ece"

python -m data_utils.domain_shift --train_datasets="sst5, rte" --unseen_datasets="medqa, wikitoxic" --gpu_id=1
python -m analysis.task_coverage --train_llms="meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, openai/gpt-oss-20b" --train_datasets="snli, sst5, rte, agnews, trec, toxic_chat, goemotions, dbpedia_l2, newsgroups, yahoo_answers" --unseen_llms="microsoft/Phi-4-mini-instruct" --unseen_datasets="amazon_counterfactual, commonsense_qa, massive_intent, medqa, metatool, when2call, wikitoxic, wildguard" --shots_start=5 --shots_end=15 --label_augment --temp_augment
Qwen/Qwen3.5-9B; google/gemma-4-E4B-it
