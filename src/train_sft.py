import click
import torch
from trainers import SFTTrainer
from src.gpt import GPT
from src.dataset import EYLSFTStaticDataset
from src.configs import get_configs

# Avoid GPU version conflict (For Kaggle GPU only). Comment below two lines if you use local machine in order to speed up training.
import torch._dynamo.config
torch._dynamo.config.suppress_errors = True

def train(pretrain, batch_size, exp_name):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
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
    train_ds = EYLSFTStaticDataset(block_size=1024,
                                   split='train',
                                   max_examples=None,
                                   tokenizer_name="tiktoken/gpt2")
    test_ds = EYLSFTStaticDataset(block_size=1024,
                                  split='test',
                                  max_examples=None,
                                  tokenizer_name="tiktoken/gpt2")
    
    trainer = SFTTrainer(cfg, device, model, train_ds, test_ds)
    trainer.fit()


@click.command()
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
def main(pretrain, batch_size, exp_name):
    torch.manual_seed(1234)
    train(pretrain, batch_size, exp_name)


if __name__ == "__main__":
    main()
