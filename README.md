# PARM 

Baijiong Lin, Weisen Jiang, Yuancheng Xu, Hao Chen, and Ying-Cong Chen. PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model. In *ICML*, 2025.

## Installation
Our code is based on [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft) for training and [Model_Arithmetic](https://github.com/eth-sri/language-model-arithmetic) for inference. 
```
conda create -n parm python=3.10
conda activate parm

cd language-model-arithmetic/
pip install -e .

cd ../peft/
pip install -e .

conda install -c nvidia cuda-compiler

cd ..
git clone https://github.com/PKU-Alignment/safe-rlhf.git
cd safe-rlhf
pip install .

cd ..
pip install -r requirements.txt
```

## Preparing Data
```
cd code/data
python relabel.py
```

## Training
```
cd code/training
bash run.sh
```

## Evaluation
```
cd code/evaluation
python generate_outputs.py --model_parm_both_name_or_path /path --alpha_helpfulness 0.5 --alpha_harmlessness 0.5
python compute_reward.py --path /path
```

## Acknowledgement
This codebase is heavily based on [GenARM](https://github.com/Yuancheng-Xu/GenARM).

## Citation
If you find this work/code useful for your research, please cite the following:
```
@inproceedings{lin2025parm,
  title={{PARM}: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model},
  author={Lin, Baijiong and Jiang, Weisen and Xu, Yuancheng and Chen, Hao and Chen, Ying-Cong},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

