Official implementation for our AACL 2026 paper [_"Towards Universal Calibration for In-Context Classification"_](https://arxiv.org/abs/2610.17096).

## Dependencies

| Package | Version |
|---|---|
| [msgspec](https://pypi.org/project/msgspec/) | `0.19.0` |
| [torch](https://pypi.org/project/torch/) | `2.10.0` |
| [vllm](https://pypi.org/project/vllm/) | `0.19.0` |
| [datasets](https://pypi.org/project/datasets/) | `4.8.4` |
| [sentence-transformers](https://pypi.org/project/sentence-transformers/) | `5.3.0` |
| [google-auth](https://pypi.org/project/google-auth/) | `2.49.1` |
| [google-genai](https://pypi.org/project/google-genai/) | `1.70.0` |
| [relplot](https://pypi.org/project/relplot/) | `1.0.1` |

```relplot``` helps calculate smoothECE and ```msgspec``` helps fast loading of the calibration training/inference datasets (if you happen to generate them) 

### Installation

```bash
pip install -r requirements.txt
```

<details>
<summary>requirements.txt</summary>

```text
msgspec==0.19.0
torch==2.10.0
vllm==0.19.0
datasets==4.8.4
sentence-transformers==5.3.0
google-auth==2.49.1
google-genai==1.70.0
relplot==1.0.1
```

</details>

Please generate the classification datasets before testing calibration (calibrator provided in the repo)

## Run classification with calibrator

The following command tests calibration for the specified shot counts, over a batch of 300 queries per seed, averaged over 10 seeds. 
TF denotes the transformer calibrator, ICT and FS_ICT refer to temperature scaling derived from predictions on in context examples.  

```bash
python run_classification.py --model="" --dataset="commonsense_qa"\
--all_shots="5, 6, 8, 10, 12"  --num_seeds=10 --subsample_test_set=300\
--sampling_strategy="entropy" --entropy_levels="random" --gpu_ids=0\
--calibration="TF, ICT, FS_ICT" --calibrator_name="calibrator"
```

## Generate Classification Dataset
By default embeddinggemma-300m is used to generate embeddings for the queries/sentences at default embedding dimension of 768. 
```bash
python --model="google/embeddinggemma-300m" --dataset="commonsense_qa" --dim=768 --gpu_id=0
```

## If your work builds on our paper, please cite our work: 

```bibtex
@misc{dhar2026universalcalibrationincontextclassification,
      title={Towards Universal Calibration for In-Context Classification}, 
      author={Awik Dhar and Sunita Sarawagi and Arpit Agarwal},
      year={2026},
      eprint={2610.17096},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.17096}, 
}
```

