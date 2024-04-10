import click
import torch
import copy
import os
from trainers import DPOTrainer
from configs import get_configs
from gpt import GPTActor, GPTRewardModel, GPTCritic, GPT
from dataset import DahoasSFTStaticPromptsDataset, RLHFDataset


def train(pretrain, batch_size, exp_name):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # cfg = get_configs("gpt2-medium/dropout")
    cfg = get_configs("gpt2-medium/dropout")
    cfg.max_steps = cfg.max_steps // batch_size
    cfg.max_cumulate_iter = cfg.max_cumulate_iter // batch_size
    cfg.eval_set_size = cfg.eval_set_size // batch_size  
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    assert pretrain == "huggingface"
    cfg.exp_name = exp_name

    # model_cache_path = f"./{cfg.model_name}"
    # if os.path.exists(model_cache_path):
    #     model = GPT.from_pretrained(model_cache_path)
    # model = GPT.from_pretrained(cfg)
    # ref_model = GPT.from_pretrained(cfg)

    # model.load_state_dict(torch.load('./runs/sft_sft_experiment_202311290742/sft_sft_experiment_202311290742_final.pt')['model_state_dict'])
    # ref_model.load_state_dict(torch.load('./runs/sft_sft_experiment_202311290742/sft_sft_experiment_202311290742_final.pt'))

    # train_ds = RLHFDataset(block_size=256, split="train", max_examples=None, tokenizer_name="tiktoken/gpt2")
    test_ds = RLHFDataset(block_size=256, split="test", max_examples=None, tokenizer_name="tiktoken/gpt2")

    sample_x, sample_attn_mask = test_ds[0]
    first_zero_index = sample_attn_mask.index(0)
    print("First zero index:", first_zero_index)
    