import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from vit import Transformer, FeedForward

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
        decoder_dim_head = 64
    ):
        super().__init__()

        self.dim = dim

        self.input_projection = input_projection

        self.projectors = nn.ModuleList([nn.Linear(input_projection, dim) for _ in range(6)])
        
        self.modality_identifiers = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(6)])

        self.register_tokens = nn.Parameter(torch.randn(2, dim))

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.num_modalities = 6
        self.num_register_tokens = 2

        self.encoder = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
        num_patches = self.num_modalities + self.num_register_tokens
        
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        #self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if dim != decoder_dim else nn.Identity()
        self.decoder_projections = nn.ModuleList([FeedForward(decoder_dim, decoder_dim) for _ in range(6)])

        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)

        self.to_embeddings = nn.ModuleList([nn.Linear(dim, input_projection) for _ in range(6)])

    def forward(self, modalities, audio_flag=0):
        device = modalities.device

        # get patches

        batch, num_patches, *_ = modalities.shape

        tokens = torch.zeros(batch, num_patches, self.dim, device = device)
        for i, projector in enumerate(self.projectors):
            tokens[:, i] = projector(modalities[:, i])

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        if torch.rand(1) < 0.9:
            num_masked = 5
        else:
            num_masked = 4

        if audio_flag:
            rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        else:
            rand_indices = torch.rand(batch, num_patches-1, device = device).argsort(dim = -1)
            masked_indices, unmasked_indices = torch.cat((rand_indices[:, :(num_masked-1)], torch.ones(batch, 1, device=device, dtype=torch.long)*5), dim=-1), rand_indices[:, (num_masked-1):]

        # mask the tokens

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens[batch_range, masked_indices] = self.mask_token

        # add modality identifiers
        for i, modality_identifier in enumerate(self.modality_identifiers):
            tokens[:, i] += modality_identifier

        # add register tokens
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)
        tokens = torch.cat((register_tokens, tokens), dim = 1)

        # attend with vision transformer

        encoded_tokens = self.encoder(tokens)

        mu = encoded_tokens[:, 0]
        logvar = encoded_tokens[:, 1]

        sample = mu + torch.exp(logvar/2) * torch.randn_like(logvar)

        sample = repeat(sample, 'b d -> b n d', n=6)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        # decoder_tokens = self.enc_to_dec(sample)

        # concat the masked tokens to the decoder tokens and attend with decoder

        # for i, modality_identifier in enumerate(self.modality_identifiers):
        #     decoder_tokens[:, i] += modality_identifier
        
        decoder_tokens = torch.zeros(batch, num_patches, self.dim, device = device)

        for i, decoder_projection in enumerate(self.decoder_projections):
            decoder_tokens[:, i] = decoder_projection(sample[:, i])
        
        for i, modality_identifier in enumerate(self.modality_identifiers):
            decoder_tokens[:, i] += modality_identifier

        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        #modality_tokens = decoded_tokens[:, self.num_register_tokens:]
        modality_tokens = decoded_tokens

        pred_embeddings = torch.zeros(batch, self.num_modalities, self.input_projection, device = device)
        for i, to_embedding in enumerate(self.to_embeddings):
            pred_embeddings[:, i] = to_embedding(modality_tokens[:, i])
        pred_embeddings = torch.nn.functional.normalize(pred_embeddings, dim=-1)

        # calculate reconstruction loss
        if audio_flag:
            recon_loss = F.mse_loss(pred_embeddings, modalities, reduction='sum') / batch
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # print(recon_loss, kl_loss)
            return recon_loss + 0.0001  * kl_loss
            
        else:
            recon_loss = F.mse_loss(pred_embeddings[:, :-1], modalities[:, :-1], reduction='sum') / batch
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # print(recon_loss, kl_loss)
            return recon_loss + 0.0001  * kl_loss
    
    def forward_inference(self, modalities, modality_mask):
        device = modalities.device
        batch, num_patches, *_ = modalities.shape
        tokens = torch.zeros(batch, num_patches, self.dim, device = device)
        for i, projector in enumerate(self.projectors):
            tokens[:, i] = projector(modalities[:, i])
        
        # add modality identifiers
        batch_range = torch.arange(batch, device = device)[:, None]
        modality_mask = repeat(modality_mask, 'n -> b n', b = batch)

        tokens[batch_range, modality_mask] = self.mask_token

        for i, modality_identifier in enumerate(self.modality_identifiers):
            tokens[:, i] += modality_identifier
        
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)
        tokens = torch.cat((register_tokens, tokens), dim = 1)

        encoded_tokens = self.encoder(tokens)[:, :2]

        return encoded_tokens
    
    def forward_decode(self, modalities, modality_mask):
        device = modalities.device
        batch, num_patches, *_ = modalities.shape
        tokens = torch.zeros(batch, num_patches, self.dim, device = device)
        for i, projector in enumerate(self.projectors):
            tokens[:, i] = projector(modalities[:, i])
        
        # add modality identifiers
        batch_range = torch.arange(batch, device = device)[:, None]
        modality_mask = repeat(modality_mask, 'n -> b n', b = batch)

        tokens[batch_range, modality_mask] = self.mask_token

        for i, modality_identifier in enumerate(self.modality_identifiers):
            tokens[:, i] += modality_identifier
        
        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = batch)
        tokens = torch.cat((register_tokens, tokens), dim = 1)

        encoded_tokens = self.encoder(tokens)

        mu = encoded_tokens[:, 0]
        logvar = encoded_tokens[:, 1]

        sample = mu + torch.exp(logvar/2) * torch.randn_like(logvar)

        sample = repeat(sample, 'b d -> b n d', n=6)

        decoder_tokens = torch.zeros(batch, num_patches, self.dim, device = device)

        for i, decoder_projection in enumerate(self.decoder_projections):
            decoder_tokens[:, i] = decoder_projection(sample[:, i])
        
        for i, modality_identifier in enumerate(self.modality_identifiers):
            decoder_tokens[:, i] += modality_identifier

        decoded_tokens = self.decoder(decoder_tokens)

        modality_tokens = decoded_tokens

        pred_embeddings = torch.zeros(batch, self.num_modalities, self.input_projection, device = device)
        for i, to_embedding in enumerate(self.to_embeddings):
            pred_embeddings[:, i] = to_embedding(modality_tokens[:, i])
        pred_embeddings = torch.nn.functional.normalize(pred_embeddings, dim=-1)

        return pred_embeddings


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