import torch
import torch.nn.functional as F


def masked_mse_loss_for_mutation(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the masked MSE loss, supporting vector targets.
    """
    if mask.dim() == input.dim() - 1:
        mask = mask.unsqueeze(-1)
    num_masked_elements = mask.sum()
    if input.dim() > mask.dim(): # 兼容旧的标量输入情况
         num_masked_elements = mask.sum() * input.shape[-1]

    if num_masked_elements == 0:
        return torch.tensor(0.0, device=input.device)

    loss = F.mse_loss(input, target, reduction="none") # 计算每个元素的平方误差
    masked_loss = loss * mask # 将未被掩码位置的loss置为0
    
    return masked_loss.sum() / num_masked_elements

def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()
