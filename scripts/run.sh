python run_classification.py --model="meta-llama/Llama-3.1-8B-Instruct" --dataset="banking77" --all_shots="8" \
 --approx --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="rand"  --subsample_test_set=100 --approx  --bs=1 --gpu_id=0 --api_num_log_prob=1000 \
 --calibration="TC" --tc_input_dim=13 --approx --calibrator_model_path="./calibration/models/meta-llama_Llama-3.1-8B-Instruct/snli_sst5_rte_agnews_trec/calibrator"

python -m calibration.generate_calibration_dataset --model="meta-llama/Llama-3.1-8B-Instruct" --dataset="metatool" --train_size=0 --test_size=1000 --num_shots=8 --gpu_id=0 --api_num_log_prob=1000
# python run_classification.py --model="google/gemma-3-12b-it" dataset="snli" --all_shots="8"  --approx --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="rand" --gpu_id=1

python -m calibration.train --model="meta-llama/Llama-3.1-8B-Instruct" --datasets="snli, sst5, rte, agnews, trec" --iterations=40000 --batch_size=16 --eval_iter=400 --lr=0.00001

python -m calibration.eval --model="meta-llama/Llama-3.1-8B-Instruct" --dataset="qqp" --model_path="./calibration/models/meta-llama_Llama-3.1-8B-Instruct/snli_sst5_rte_agnews_trec/calibrator" --gpu_id=0

for d in */; do
  printf "%s %s\n" "$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)" "$d"
done | sort -nr | awk '{print strftime("%Y-%m-%d %H:%M:%S", $1), $2}'