import click
import torch
from transformers import GPT2Tokenizer

from trainers import SFTTrainer
from src.gpt import GPT
from src.dataset import EYLSFTStaticDataset, SFTDataset, SupervisedDataset
from src.configs import get_configs
from tokenizer import TiktokenTokenizer
import utils
import random

# Avoid GPU version conflict (For Kaggle GPU only). Comment below two lines if you use local machine in order to speed up training.
import torch._dynamo.config
torch._dynamo.config.suppress_errors = True

def train(pretrain, batch_size, exp_name):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    cfg = get_configs("gpt2-medium") # change this line to select different models
    # cfg = get_configs("gpt2") # change this line to select different models
    
    # cfg.max_steps = 200000 // batch_size
    cfg.max_steps = cfg.max_steps // batch_size
    cfg.max_cumulate_iter = cfg.max_cumulate_iter // batch_size
    cfg.eval_set_size = cfg.eval_set_size // batch_size
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    assert pretrain == "huggingface" # make sure the pretrained model is in the format of huggingface.
    cfg.exp_name = exp_name

    # load the pretrained GPT model based on the configuration
    model = GPT.from_pretrained(cfg)
    
    # load SFT dataset
    # train_ds = EYLSFTStaticDataset(block_size=1024,
    #                                split='train',
    #                                max_examples=None,
    #                                tokenizer_name="tiktoken/gpt2")
    # test_ds = EYLSFTStaticDataset(block_size=1024,
    #                               split='test',
    #                               max_examples=None,
    #                               tokenizer_name="tiktoken/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained('../checkpoints/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    list_data_dict = utils.jload(f"./alpaca_data.json")
    random.shuffle(list_data_dict)
    # 计算90%的索引位置
    split_index = int(0.9 * len(list_data_dict))

    # 使用列表切片分割数据集
    train_data = list_data_dict[:split_index]  # 取随机的90%作为训练集
    test_data = list_data_dict[split_index:]
    train_ds = SupervisedDataset(data=train_data, tokenizer=tokenizer)
    test_ds = SupervisedDataset(data=test_data, tokenizer=tokenizer)
    # test_ds = SFTDataset(block_size=1024,
    #                               split='test',
    #                               max_examples=None,
    #                               tokenizer_name="tiktoken/gpt2")
    
    trainer = SFTTrainer(cfg, device, model, train_ds, test_ds, tokenizer=tokenizer)
    trainer.fit()


@click.command()
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=2)
@click.option('--exp-name', '-n', default="default")
def main(pretrain, batch_size, exp_name):
    torch.manual_seed(1234)
    train(pretrain, batch_size, exp_name)


if __name__ == "__main__":
    main()
