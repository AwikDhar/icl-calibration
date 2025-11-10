python run_classification.py --model="meta-llama/Llama-3.1-8B-Instruct" --dataset="banking77" --all_shots="8" \
 --approx --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="rand"  --subsample_test_set=100 --approx  --bs=1 --gpu_id=0 --api_num_log_prob=1000 \
 --calibration="TC" --tc_input_dim=13 --approx --calibrator_model_path="./calibration/models/meta-llama_Llama-3.1-8B-Instruct/snli_sst5_rte_agnews_trec/calibrator"

python -m calibration.generate_calibration_dataset --model="meta-llama/Llama-3.1-8B-Instruct" --dataset="sst5" --train_size=5000 --test_size=1000 --num_shots=8 --gpu_id=0 --api_num_log_prob=1000
# python run_classification.py --model="google/gemma-3-12b-it" dataset="snli" --all_shots="8"  --approx --num_seeds=10 --sampling_strategy="entropy" --entropy_levels="rand" --gpu_id=1

python -m calibration.train --models="meta-llama/Llama-3.1-8B-Instruct" --datasets="snli, sst5, rte, agnews, trec, dbpedia_l2, toxic_chat" --iterations=40000 --batch_size=8 --eval_iter=400 --lr=0.00001

python -m calibration.eval --models="meta-llama/Llama-3.1-8B-Instruct" --datasets="qqp" --model_path="./calibration/models/meta-llama_Llama-3.1-8B-Instruct/snli_sst5_rte_agnews_trec/calibrator" --gpu_id=0

for d in */; do
  printf "%s %s\n" "$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)" "$d"
done | sort -nr | awk '{print strftime("%Y-%m-%d %H:%M:%S", $1), $2}'

find . -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n'

find . -type f -printf '%T@ %p\n' | sort -n | tail -20 | cut -d' ' -f2-
