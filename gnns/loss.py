import torch
import torch.nn.functional as F
from torch import Tensor


class MSELoss:
    def __init__(self,) -> None:
        pass

    def forward(self, h_input: Tensor, h_target: Tensor):
        # input_dim = h_input.shape[1]
        # target_dim = h_target.shape[1]
        # if input_dim == target_dim + 1:  
        #     # if predict cost and deadend
        #     h_input = h_input[:,:-1]
        # assert h_input.shape == h_target.shape
        h_loss = F.mse_loss(h_input, h_target)
        return h_loss

class BCELoss:
    def __init__(self,) -> None:
        pass

    def forward(self, h_input: Tensor, h_target: Tensor):
        # h_input = h_input[:,-1]
        # assert h_input.shape == h_target.shape
        h_loss = F.binary_cross_entropy_with_logits(h_input, h_target)
        return h_loss

class InfoNCELoss:
    """InfoNCE loss"""
    def __init__(self, temperature=0.1, scale_by_temperature=False):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, anchor_emb: torch.Tensor, positive_emb: torch.Tensor):
        """
        anchor_emb: [batch_size, embedding_dim]
        positive_emb: [batch_size, embedding_dim]
        """
        batch_size = anchor_emb.size(0)

        # 拼接anchor和positive
        all_emb = torch.cat([anchor_emb, positive_emb], dim=0)
        all_emb = F.normalize(all_emb, dim=1)

        # 计算anchor与所有样本的相似度矩阵
        sim = torch.matmul(all_emb[:batch_size], all_emb.T) / self.temperature  # [bs, 2*bs]

        # 每个anchor的正样本在拼接后的位置为 batch_size + i
        labels = torch.arange(batch_size, device=all_emb.device) + batch_size

        # 不计算anchor与自身的相似度
        mask = torch.arange(batch_size, device=all_emb.device)
        sim[torch.arange(batch_size), mask] = -1e9  

        # 交叉熵损失
        loss = F.cross_entropy(sim, labels)

        return loss
    
class CombinedLoss:
    def __init__(self, alpha=1, temperature=0.1, scale_by_temperature=True) -> None:
        self.alpha = alpha
        self.mse_loss_fn = MSELoss()
        self.supcon_loss_fn = InfoNCELoss(temperature, scale_by_temperature)

    def forward(self, embeddings: Tensor, h_input: Tensor, h_target: Tensor):
        mse_loss = self.mse_loss_fn.forward(h_input, h_target)
        supcon_loss = self.supcon_loss_fn.forward(embeddings, h_target)
        combined_loss = mse_loss + self.alpha * supcon_loss 
        print(f"MSE Loss: {mse_loss.item()}, SupCon Loss: {supcon_loss.item()}")
        return combined_loss
