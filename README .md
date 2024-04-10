Fine-Tuning Large Language Models — A Case Study from GPT-2<a name="TOP"></a>
===================

- - - - 
# 1.introduction #
Large Language Models (LLMs) have demonstrated remarkable capabilities in general tasks and garnered significant attention in recent years.  Efforts to align these models with human preferences have included methods like Supervised Fine-Tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF).  
In our study, we take GPT-2 as a representative model to explore and suggest several strategies to lower the costs associated with fine-tuning LLMs.

## install ##
```shell
cd ./src
conda create -n ML
conda activate ML
pip install -r requirements.txt
```

#  Base Task: Supervised Fine-Tuning (SFT) #
our base task was to implement Supervised Fine-Tuning (SFT) for GPT-2 using the "HH-RLHF" dataset to facilitate its dialogue ability with human.
- Dataset Preparation
- Model finetuning Process: Using GPT-2 as the foundation model, we trained and fine-tuned it on the prepared dataset. In our training process, we've adopted an accumulated gradient update strategy. This approach involves updating the model parameters only after a certain number of forward and backward passes, which speeds up the training process.  `trains.py`
- Periodic Assessment：
To ensure our model is learning effectively, we perform periodic evaluations after every set number of iterations. The best checkpoint (lowest evaluation loss) will be stored in the out_dir directory.
At the end, You can then run the code in `train_sft.py`
- Evaluation: we compared average reward scores from four scenarios: responses by the SFT model, the original pre-trained model, and actual responses from the test and training sets. you can run the code in `evaluate.py`

Model       | Average reward scores
------------- | -------------
SFT model     | -3.19
Pretrained model  | -3.37
Test set answers  | -1.41

```shell
python train_sft.py --exp-name sft_experiment --batch-size 4 --max-cumulate-iter 1024
```
Before running the evaluation code, please check the path pf the SFT model. Remember to replace the path in  `./src/evaluate.py` with your own path of the SFT model.
```shell
python evaluate.py
```



# exploration #
## LoRA ##
In the LoRA section, we conducted experiments with various LoRA ranks to analyze their impact. The experiment results offer insights into the optimal configuration of LoRA for our model.

```shell
python train_sft_lora.py --exp-name sft_experiment --batch-size 1 -t 1 --lora-rank 8 --lora-bias all
```

## DPO & CDPO Comparative Analysis ##
We conducted an experimental comparison between Direct Preference Optimization (DPO) and Conservative Direct Preference Optimization (CDPO). Our goal is to evaluate the loss and classification accuracy of each method on the "HH-RLHF" dataset.
- Experimental Setup: We also utilized GPT-2 medium SFT model with gradient accumulation for larger batch size processing, and we employed the AdamW optimizer for training.
- Assessment: Both training and evaluation losses were measured, along with binary classification accuracy, to provide a balanced view of the model's performance.
- Findings: The CDPO algorithm demonstrated a significant reduction in variance and an accelerated rate of convergence compared to DPO. Notably, CDPO achieved the desired accuracy in half the GPU time required by DPO.
- Loss vs. Accuracy: We noted that while DPO consistently improved accuracy, the evaluation loss did not always correlate with accuracy improvement, suggesting that DPO loss may offer a biased performance estimate.
  ![](pic/DPO&CDPO.png)

Before running the code, please check the path pf the SFT model. Remember to replace the path in line 38 of `./src/train_dpo.py` with your own path of the SFT model.
```shell
python train_dpo.py --exp-name dpo_experiment --batch-size 2 --max-cumulate-iter 2048 --eval-interval 0
python train_dpo.py --exp-name dpo_experiment --batch-size 2 --max-cumulate-iter 2048 --eval-interval 0 --conservative-loss True
```

## AGMA ##
we evaluated AGMA on the high-variance stochastic optimization problem posed by the DPO task.
- Demonstrated superior convergence rate of AGMA over AdamW, and confirmed AGMA's lower memory cost comparable to Adam and AdamW.
- Explored various configurations including step size strategies and updates styles.
- Investigated the impact of step size reduction and parameter perturbation but we haven't actually figured out the cause of stagnation.

Before running the code, please check the path pf the SFT model. Remember to replace the path in line 38 of `./src/train_dpo.py` with your own path of the SFT model.
```shell
python train_dpo.py --exp-name dpo_experiment --batch-size 2 --max-cumulate-iter 0 --eval-interval 1024 --optimizer AGMA
```







