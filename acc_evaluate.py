import click
import torch
from torch.utils.data import DataLoader
import copy
import os
from trainers import DPOTrainer
from configs import get_configs
from gpt import GPTActor, GPTRewardModel, GPTCritic, GPT
from dataset import DahoasSFTStaticPromptsDataset, RLHFDataset
from tqdm import tqdm



@torch.no_grad()
def acc_eval(pretrain, batch_size, exp_name, max_cumulate_iter, conservative_loss, contrastive_loss):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # cfg = get_configs("gpt2-medium/dropout")
    cfg = get_configs("gpt2-medium/dropout")
    cfg.max_steps = cfg.max_steps // batch_size
    cfg.max_cumulate_iter = cfg.max_cumulate_iter // batch_size
    cfg.eval_set_size = cfg.eval_set_size // batch_size  
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    # cfg.max_cumulate_iter = max_cumulate_iter
    cfg.conservative_loss = conservative_loss
    cfg.contrastive_loss = contrastive_loss
    assert not (cfg.conservative_loss and cfg.contrastive_loss)

    assert pretrain == "huggingface"
    
    model = GPT.from_pretrained(cfg)
    state_dict = torch.load('./runs/sft_sft_experiment_202311290742/sft_sft_experiment_202311290742_final.pt')['model_state_dict']
    # state_dict = torch.load('./runs/sft_dpo_experiment_202311301858/sft_dpo_experiment_202311301858_final.pt')['model_state_dict']
    model.load_state_dict(state_dict)
    test_ds = RLHFDataset(block_size=256, split="test", max_examples=None, tokenizer_name="tiktoken/gpt2")

    test_dataloader = DataLoader(test_ds,
                       batch_size=cfg.batch_size,
                       num_workers=min(6,cfg.batch_size),
                    #    shuffle=True,
                       pin_memory=True)
    model.eval()
    model.to(device)
    acc_list, B_list = [], []
    batch_same_list = []
    for x,attn_mask in tqdm(test_dataloader):
        # orig_x = copy.deepcopy(x)
        # break
        B, _, T = x.size()
        x, attn_mask = x.reshape(-1, T), attn_mask.reshape(-1, T)
        x, attn_mask = x.to(device), attn_mask.to(device)
        logits = model(x, attention_mask=attn_mask)
        _, _, C = logits.shape
        logits = logits.view(-1,C)

        logits, x = logits.view(B, 2, T, C), x.view(B, 2, T)
        logits, x = logits[:,:,:-1,:], x[:,:,1:]
        logits_pos, logits_neg = torch.split(logits, 1, dim=1)
        logits_pos, logits_neg = logits_pos.squeeze(1), logits_neg.squeeze(1)
        logits_pos, logits_neg = logits_pos.reshape(B,T-1,C), logits_neg.reshape(B,T-1,C)
        x_pos, x_neg = torch.split(x, 1, dim=1)
        x_pos, x_neg = x_pos.squeeze(1), x_neg.squeeze(1)
        x_pos, x_neg = x_pos.reshape(B,T-1), x_neg.reshape(B,T-1)
        pred_prob_pos = torch.softmax(logits_pos.squeeze(1), dim=-1)
        pred_prob_neg = torch.softmax(logits_neg.squeeze(1), dim=-1)
        orig_prob_pos = pred_prob_pos.clone()
        # pred_prob_pos1, pred_prob_pos2 = torch.split(pred_prob_pos, 1, dim=0)
        # batch_same_list.append(torch.equal(pred_prob_pos1, pred_prob_pos2))
        pred_prob_pos = torch.gather(pred_prob_pos, dim=-1, index=x_pos.unsqueeze(-1)).squeeze(-1)
        pred_prob_neg = torch.gather(pred_prob_neg, dim=-1, index=x_neg.unsqueeze(-1)).squeeze(-1)
        # pred_prob_pos1, pred_prob_pos2 = torch.split(pred_prob_pos, 1, dim=0)
        # batch_same_list.append(torch.equal(pred_prob_pos1, pred_prob_pos2))
        # print(pred_prob_pos.size())
        # sentence_prob_pos = torch.prod(pred_prob_pos, dim=-1)
        # sentence_prob_neg = torch.prod(pred_prob_neg, dim=-1)
        sentence_log_prob_pos = torch.sum(torch.log(pred_prob_pos), dim=-1)
        sentence_log_prob_neg = torch.sum(torch.log(pred_prob_neg), dim=-1)
        # print(sentence_log_prob_pos.size())
        assert B == sentence_log_prob_pos.size(0)

        acc = (sentence_log_prob_pos > sentence_log_prob_neg).float().mean().item()
        acc_list.append(acc)
        B_list.append(B)
        # break
    acc = sum([acc_list[i] * B_list[i] for i in range(len(acc_list))]) / sum(B_list)
    # acc = sum(acc_list) / len(acc_list)
    print(f"acc: {acc}")
    

@click.command()
@click.option('--strategy', '-s')
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=2)
@click.option('--exp-name', '-n', default="default")
@click.option('--max-cumulate-iter', '-b', default=1024)
@click.option('--conservative-loss', '-c', default=False)
@click.option('--contrastive-loss', '-c', default=False)
def main(strategy, pretrain, batch_size, exp_name, max_cumulate_iter, conservative_loss, contrastive_loss):
    torch.manual_seed(1234)
    acc_eval(pretrain, batch_size, exp_name, max_cumulate_iter, conservative_loss, contrastive_loss)

if __name__ == "__main__":
    main()
    # B,T,C=2,3,4
    # logits = torch.arange(B*T*C).view(B,T,C)
    # x = torch.arange(B*T).view(B,T)
    # x = torch.remainder(x,C)
    # print(logits)
    # print(x)
    # pred_prob = torch.gather(logits, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
    # print(pred_prob)
    