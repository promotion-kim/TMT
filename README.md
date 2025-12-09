# Tchebycheff scalarized Multi-objective Test-time alignment (TMT) 

To capture non-convex Pareto frontiers, we employ Tchebycheff (TCH) and Smooth Tchebycheff (STCH) scalarization in the optimization process.

## Installation
Our code is based on [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft) for training and [Model_Arithmetic](https://github.com/eth-sri/language-model-arithmetic) for inference (test-ti). 
```
conda create -n tmt python=3.10
conda activate tmt

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
cd code/data # or code/data/hh
python relabel.py
python relabel_hh.py
```

## Training
```
cd code/training
bash run.sh
```

## Evaluation
```
cd code/evaluation # or code/hh/evaluation
bash generation.sh
bash compute_reward.sh
```
For Pareto frontier visualization and metric comparisons, see the pareto.ipynb file.

## Acknowledgement
This codebase is heavily based on [PARM](https://github.com/Baijiong-Lin/PARM).
