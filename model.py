import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
import warnings
from linear_vit import Transformer, FeedForward
import numpy as np


def create_pairwise_mask(labels):
    num_samples = len(labels)
    pairwise_mask = torch.zeros(num_samples, num_samples).to(labels.device)

    for i in range(num_samples):
        pairwise_mask[i, :] = torch.all(labels == labels[i], dim=-1)

    return pairwise_mask

def mse_contrastive_loss(pred_embeds, gt_embeds, label):
    """
    pred_embeds: [num_modalities, batch_size, embed_dim]
    gt_embeds: [num_modalities, batch_size, embed_dim]
    label: [batch_size, embed_dim]
    """
    dist = torch.cdist(pred_embeds, gt_embeds)
    dist = -5 * dist + 5
    gt = repeat(create_pairwise_mask(label), 'b m -> n b m', n=pred_embeds.shape[0])
    # dist2 = torch.cdist(pred_embeds, pred_embeds)
    # dist2 = -5 * dist2 + 5
    # loss = F.binary_cross_entropy_with_logits(dist, gt)
    # import code; code.interact(local=dict(globals(), **locals()))
    loss = (-torch.log(dist.softmax(dim=-1)+1e-6)*gt).sum(-1).mean(-1).mean()
    # loss2 = (-torch.log(dist2.softmax(dim=-1)+1e-6)*gt).sum(-1).mean(-1).mean()
    return loss


class M3E(nn.Module):
    def __init__(
        self,
        *,
        input_projection=512,
        dim=512,
        depth=6,
        heads=8,
        dim_head=64,
        mlp_dim=2048,
        decoder_dim=512,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        rand_noise_prob = 0.1,
        contrastive_loss = False
    ):
        super().__init__()

        # self.linear = nn.Linear(input_projection, input_projection)
        self.projector = nn.ModuleList([nn.Linear(input_projection, dim) for _ in range(5)])
        self.modality_identifier = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(5)])
        self.register_tokens = nn.Parameter(torch.randn(2, dim))
        self.enc = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.cls = nn.Parameter(torch.randn(dim))
        # self.cls.requires_grad = False
        self.linear = nn.ModuleList([nn.Linear(dim, input_projection) for _ in range(5)])

        self.dim=dim

    def forward(self, modalities, audio_flag=0):
        device = modalities.device

        pred = torch.zeros((modalities.shape[0], 5, modalities.shape[-1]), device=device)
        if torch.rand(1) < 0.9:
            num_masked = 4
        else:
            num_masked = 3
        
        idx = np.random.choice([0, 1, 2, 3, 4], 5-num_masked, replace=False)
        x = torch.zeros((modalities.shape[0], 5-num_masked, self.dim), device=device)
        for i, ids in enumerate(idx):
            x[:, i] = self.projector[ids](modalities[:, ids]) + self.modality_identifier[ids]

        #x = self.projector[idx](modalities[:, idx])
        cls_token = repeat(self.cls, 'd -> b d', b=modalities.shape[0]).unsqueeze(1)
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b=modalities.shape[0])
        x = torch.cat((register_tokens, x), dim=1)

        x = self.enc(cls_token, x, x)[0][:, 0]
        
        for i, layer in enumerate(self.linear):
            pred[:, i] = layer(x)
        
        pred = torch.nn.functional.normalize(pred, dim=-1)

        # return F.mse_loss(pred, modalities[:, :5])
        loss = 0
        for i in range(5):
            dist = torch.cdist(pred[:, i], modalities[:, i])
            dist = -5 * dist + 5
            loss += (-torch.log(dist.softmax(-1) + 1e-6) * torch.eye(dist.shape[0], device=device)).sum(-1).mean()
        return loss / 5

    
    def forward_inference(self, modalities, modality_mask):
        device = modalities.device

        pred = torch.zeros((modalities.shape[0], 5, modalities.shape[-1]), device=device)
        
        # idx = np.random.choice([0, 1, 2, 3, 4], 5-num_masked, replace=False)
        idx = [0, 1]
        x = torch.zeros((modalities.shape[0], 2, self.dim), device=device)
        for i, ids in enumerate(idx):
            x[:, i] = self.projector[ids](modalities[:, ids]) + self.modality_identifier[ids]

        #x = self.projector[idx](modalities[:, idx])
        cls_token = repeat(self.cls, 'd -> b d', b=modalities.shape[0]).unsqueeze(1)
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b=modalities.shape[0])
        x = torch.cat((register_tokens, x), dim=1)

        x = self.enc(cls_token, x, x)[0][:, 0]
        
        for i, layer in enumerate(self.linear):
            pred[:, i] = layer(x)

        return torch.nn.functional.normalize(pred, dim=-1)
    
    def forward_decode(self, modalities, modality_mask, n_samples=1):
        device = modalities.device
        batch, num_patches, *_ = modalities.shape
        tokens = torch.zeros(batch, num_patches, self.dim, device = device)
        for i, projector in enumerate(self.projectors):
            tokens[:, i] = projector(modalities[:, i])
        
        # add modality identifiers
        batch_range = torch.arange(batch, device = device)[:, None]
        # modality_mask = repeat(modality_mask, 'n -> b n', b = batch)

        # tokens[batch_range, modality_mask] = self.mask_token

        for i, modality_identifier in enumerate(self.modality_identifiers):
            tokens[:, i] += modality_identifier
        
        tokens = tokens[batch_range, modality_mask]
        
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)
        tokens = torch.cat((register_tokens, tokens), dim = 1)

        encoded_tokens = self.encoder(tokens)

        mu = torch.nn.functional.normalize(encoded_tokens[:, 0], dim=-1)
        logvar = encoded_tokens[:, 1]

        output_samples = torch.zeros(batch, n_samples, self.num_modalities, self.input_projection, device = device)

        for j in range(n_samples):
            sample = mu + torch.einsum('bd,b->bd',torch.exp(logvar/2), torch.randn(batch, device = device))

            sample = repeat(sample, 'b d -> b n d', n=6)

            decoder_tokens = torch.zeros(batch, num_patches, self.dim, device = device)

            for i, decoder_projection in enumerate(self.decoder_projections):
                decoder_tokens[:, i] = decoder_projection(sample[:, i])

            modality_tokens = decoder_tokens

            pred_embeddings = torch.zeros(batch, self.num_modalities, self.input_projection, device = device)
            for i, to_embedding in enumerate(self.to_embeddings):
                pred_embeddings[:, i] = to_embedding(modality_tokens[:, i])
            pred_embeddings = torch.nn.functional.normalize(pred_embeddings, dim=-1)

            output_samples[:, j] = pred_embeddings

        return output_samples


if __name__ == '__main__':
    model = M3E(input_projection=512,
                dim=512,
                depth=6,
                heads=8,
                dim_head=64,
                mlp_dim=2048,
                decoder_dim=512,
                decoder_depth=4,
                decoder_heads=8,
                decoder_dim_head=64)

    modalities = torch.randn(2, 6, 512)
    loss = model(modalities)

    # print number of parameters
    print(sum(p.numel() for p in model.parameters()))