import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import math
# import scheduler base class
from torch.optim.lr_scheduler import LRScheduler, EPOCH_DEPRECATION_WARNING, _enable_get_lr_call
# from torch.optim import Optimizer
from torch.nn.functional import cosine_similarity
import warnings
import numpy as np


class PMA(LRScheduler):
    def __init__(self, optimizer, cumulate_steps, last_epoch=-1):
        self.cumulate_steps = cumulate_steps
        self.base_beta1s = [group['betas'][0] for group in optimizer.param_groups]
        self.base_beta2s = [group['betas'][1] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)
        # self.base_beta1 = optimizer.defaults['betas'][0]
        # self.base_beta2 = optimizer.defaults['betas'][1]

    def get_lr(self):
        if self.last_epoch % self.cumulate_steps == 0 and self.last_epoch != 0:
            return [base_lr for base_lr in self.base_lrs]
        else:
            # return [base_lr * (1.0 + math.cos(math.pi * (self.last_epoch - self.cumulate_steps) / (self.max_iter - self.warmup_steps))) / 2.0 for base_lr in self.base_lrs]
            # tuning beta and alpha in adam
            return [base_lr / self.cumulate_steps**2 for base_lr in self.base_lrs]
        
    def get_betas(self):
        if self.last_epoch % self.cumulate_steps == 0 and self.last_epoch != 0:
            return [base_beta1 for base_beta1 in self.base_beta1s], [base_beta2 for base_beta2 in self.base_beta2s]
        else:
            return [base_beta1 * (self.last_epoch % self.cumulate_steps) / (self.last_epoch % self.cumulate_steps +1) for base_beta1 in self.base_beta1s], [base_beta2 * (self.last_epoch % self.cumulate_steps) / (self.last_epoch % self.cumulate_steps +1) for base_beta2 in self.base_beta2s]
    
    def step(self, epoch=None):
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                lrs = self.get_lr()
                beta1s, beta2s = self.get_betas()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    lrs = self._get_closed_form_lr()
                    raise NotImplementedError
                else:
                    lrs = self.get_lr()
                    beta1s, beta2s = self.get_betas()

        for i, (param_group, lr, data_beta1, data_beta2) in enumerate(zip(self.optimizer.param_groups, lrs, beta1s, beta2s)):
            # param_group, lr = data
            param_group['lr'] = lr
            param_group['betas'] = (data_beta1, data_beta2)
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self._last_beta1 = [group['betas'][0] for group in self.optimizer.param_groups]
        self._last_beta2 = [group['betas'][1] for group in self.optimizer.param_groups]


class WarmUpScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch > self.warmup_steps:
            return [base_lr
                    for base_lr in self.base_lrs]
        else:
            return [base_lr * self.last_epoch / self.warmup_steps
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class AGMA(Optimizer):
    """This is a self-defined optimizer based on AdamW.
        the step size is based on the consine similarity between the gradient and the momentum.
        the momentum is the exponential moving average of the gradient.
    """
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, accumulate_steps=1, weight_decay=0.01, norm_decay=0.01, variance_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, accumulate_steps=accumulate_steps, weight_decay=weight_decay, norm_decay=norm_decay, variance_decay=variance_decay)
        self.accumulate_steps = accumulate_steps
        super(AGMA, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                step_size = group['lr']
                large_step_times = state['step'] // self.accumulate_steps
                if state['step'] % self.accumulate_steps == 0 and state['step'] != 0:
                    # # pass
                    # # step_size *= ((cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2)

                    # step_size *= torch.norm(p).item() / (torch.norm(p).item() + group['norm_decay'] * torch.norm(grad).item())
                    # # sim_reg = ((cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2)*group['variance_decay'] + 1-group['variance_decay']
                    # sim_reg = ((cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2)
                    # step_size *= sim_reg
                    # sim_reg = (cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2
                    # step_size *= (1+group['variance_decay']*sim_reg)
                    # step_size *= 1/(1+group['variance_decay']*(1-sim_reg))
                    # # sim_reg = (cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2
                    # # step_size *= torch.norm(p).item() / (torch.norm(p).item() + group['weight_decay'] * (1-sim_reg) * torch.norm(grad).item())

                    exp_avg.mul_(beta1).add_(grad, alpha=(1-beta1)/self.accumulate_steps)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2)/self.accumulate_steps)
                    denom = exp_avg_sq.sqrt().add_(1e-8)
                    p.data.mul_(1 - step_size * group['weight_decay'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    # ###################################################
                    # small_step_times = state['step'] % self.accumulate_steps
                    # step_size = step_size / (large_step_times+1) / self.accumulate_steps
                    # # step_size = step_size * torch.norm(p).item() / (torch.norm(p).item() + group['norm_decay'] * torch.norm(grad).item())
                    # # sim_reg = ((cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2)*group['variance_decay'] + 1-group['variance_decay']
                    # # sim_reg = (cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2
                    # # step_size *= 1/(1+group['variance_decay']*(1-sim_reg))
                    # # sim_reg = (cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2
                    # # step_size *= sim_reg
                    
                    # beta1_, beta2_ = group['betas']

                    # # exp_avg.mul_(beta1_).add_(grad, alpha=beta1_ / self.accumulate_steps)
                    # # exp_avg_sq.mul_(beta2_).addcmul_(grad, grad, value=beta2_ / self.accumulate_steps)
                    # exp_avg.mul_((self.accumulate_steps)/(self.accumulate_steps+1)).add_(grad, alpha=beta1_ / (self.accumulate_steps+1))
                    # exp_avg_sq.mul_((self.accumulate_steps)/(self.accumulate_steps+1)).addcmul_(grad, grad, value=beta2_ / (self.accumulate_steps+1))
                    # denom = exp_avg_sq.sqrt().add_(1e-8)
                    
                    p.data.mul_(1 - step_size * group['weight_decay'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    # ###################################################
                    exp_avg.mul_(self.accumulate_steps * beta1)
                    exp_avg_sq.mul_(self.accumulate_steps * beta2)

                    # if large_step_times % 10 == 0 and large_step_times != 0:
                    #     p.data.add_(grad, alpha=-step_size/self.accumulate_steps)
                else:
                    small_step_times = state['step'] % self.accumulate_steps
                    # step_size = step_size / (large_step_times+1) / (self.accumulate_steps**2) * small_step_times * (cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2
                    # step_size = step_size / (large_step_times+1) / (self.accumulate_steps**2) * small_step_times
                    # step_size = step_size / (large_step_times+1) / self.accumulate_steps / max(small_step_times,1)
                    # step_size = step_size / (large_step_times+1) / self.accumulate_steps
                    # step_size = step_size / self.accumulate_steps
                    step_size = step_size / np.sqrt(self.accumulate_steps)
                    # step_size = step_size / (large_step_times+1) / np.sqrt(self.accumulate_steps)
                    # step_size = step_size / (self.accumulate_steps**2)

                    # step_size = step_size * torch.norm(p).item() / (torch.norm(p).item() + group['norm_decay'] * torch.norm(grad).item())
                    # sim_reg = ((cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2)*group['variance_decay'] + 1-group['variance_decay']
                    # sim_reg = (cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2
                    # step_size *= 1/(1+group['variance_decay']*(1-sim_reg))
                    # step_size *= (1+group['variance_decay']*sim_reg)
                    # sim_reg = (cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2
                    # step_size *= sim_reg
                    # # step_size = step_size / (large_step_times+1) / self.accumulate_steps * torch.norm(p).item() / (torch.norm(p).item() + group['weight_decay'] * (1-sim_reg) * torch.norm(grad).item())

                    beta1_, beta2_ = group['betas']
                    # beta1 = (small_step_times) / (small_step_times +1)
                    # beta2 = (small_step_times) / (small_step_times +1)
                    # beta1 = beta1 * (small_step_times) / (small_step_times +1)
                    # beta2 = beta2 * (small_step_times) / (small_step_times +1)
                    # exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                    # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # denom = exp_avg_sq.sqrt().add_(1e-8)
                    # p.data.mul_(1 - step_size * group['weight_decay'])
                    # p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    exp_avg.mul_((small_step_times) / (small_step_times +1)).add_(grad, alpha=(1-beta1_) / (small_step_times+1))
                    exp_avg_sq.mul_((small_step_times) / (small_step_times +1)).addcmul_(grad, grad, value=(1-beta2_) / (small_step_times+1))
                    denom = exp_avg_sq.sqrt().add_(1e-8)

                    # sim_reg = (cosine_similarity(grad.view(1, -1), exp_avg.view(1, -1)).item()+1)/2
                    # step_size *= sim_reg
                    
                    p.data.mul_(1 - step_size * group['weight_decay'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    # p.data.add_(grad, alpha=-step_size)
                    
                    # if small_step_times == self.accumulate_steps // 2:
                    #     p.data.add_(grad, value=-step_size)
        return loss