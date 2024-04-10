import click
import torch
import copy
import os
from trainers import DPOTrainer
from configs import get_configs
from gpt import GPTActor, GPTRewardModel, GPTCritic, GPT
from dataset import DahoasSFTStaticPromptsDataset, RLHFDataset


def train(pretrain, batch_size, exp_name, max_cumulate_iter, eval_interval, conservative_loss, contrastive_loss, clamp, gradient_clipping, optimizer, device="cuda:0"):
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # cfg = get_configs("gpt2-medium/dropout")
    cfg = get_configs("gpt2-medium/dropout")
    cfg.max_steps = cfg.max_steps // batch_size
    cfg.max_cumulate_iter = max_cumulate_iter // batch_size
    cfg.eval_interval = eval_interval
    cfg.eval_set_size = cfg.eval_set_size // batch_size  
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    cfg.loss_clamp = clamp
    cfg.gradient_clipping = gradient_clipping
    # cfg.max_cumulate_iter = max_cumulate_iter
    cfg.conservative_loss = conservative_loss
    cfg.contrastive_loss = contrastive_loss
    cfg.optimizer = optimizer
    assert not (cfg.conservative_loss and cfg.contrastive_loss)

    assert pretrain == "huggingface"
    cfg.exp_name = exp_name

    model_cache_path = f"./{cfg.model_name}"
    if os.path.exists(model_cache_path):
        model = GPT.from_pretrained(model_cache_path)
    model = GPT.from_pretrained(cfg)
    ref_model = GPT.from_pretrained(cfg)

    state_dict = torch.load('./runs/sft_sft_experiment_202311290742/sft_sft_experiment_202311290742_final.pt')['model_state_dict']
    # ref_state_dict = torch.load('./runs/sft_sft_experiment_202311290742/sft_sft_experiment_202311290742_final.pt')['model_state_dict']
    model.load_state_dict(state_dict)
    ref_model.load_state_dict(state_dict=state_dict)

    train_ds = RLHFDataset(block_size=256, split="train", tokenizer_name="tiktoken/gpt2")
    test_ds = RLHFDataset(block_size=256, split="test", tokenizer_name="tiktoken/gpt2")
    # train_ds = RLHFDataset(block_size=256, split="train", max_examples=cfg.max_train_example, tokenizer_name="huggingface/gpt2fast")
    # test_ds = RLHFDataset(block_size=256, split="test", max_examples=cfg.max_test_example, tokenizer_name="huggingface/gpt2fast")

    # from torch.utils.data import DataLoader
    # train_dataloader = iter(
    #         DataLoader(train_ds,
    #                    batch_size=cfg.batch_size,
    #                    num_workers=min(6,cfg.batch_size),
    #                 #    shuffle=True,
    #                    pin_memory=True))
    # for x,y in train_dataloader:
    #     print(x.size())
    #     print(y.size())
    #     B, _, T = x.size()
    #     x1 = x.reshape(-1, T)
    #     x1 = x1.reshape(B, -1, T)
    #     print(torch.equal(x, x1))
    #     break

    trainer = DPOTrainer(cfg, device, model, ref_model, train_ds, test_ds, beta=1.)
    trainer.fit()


@click.command()
@click.option('--strategy', '-s')
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="default")
@click.option('--max-cumulate-iter', '-b', default=1024)
@click.option('--eval-interval', '-e', default=0)
@click.option('--conservative-loss', '-c', default=False)
@click.option('--contrastive-loss', '-c', default=False)
@click.option('--clamp', '-c', default=False)
@click.option('--gradient-clipping', '-g', default=False)
@click.option('--device', '-d', default="cuda:0")
@click.option('--optimizer', '-o', default="AdamW")
def main(strategy, pretrain, batch_size, exp_name, max_cumulate_iter, eval_interval, conservative_loss, contrastive_loss, clamp, gradient_clipping, optimizer, device="cuda:0"):
    torch.manual_seed(1234)
    train(pretrain, batch_size, exp_name, max_cumulate_iter, eval_interval, conservative_loss, contrastive_loss, clamp, gradient_clipping, optimizer, device)



if __name__ == "__main__":
    main()
