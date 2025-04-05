import torch
import torch.nn as nn
import torchvision.models as models
try:
    from linformer import Linformer
except ImportError:
    Linformer = None  

from .tokenizer import HFTokenizer

class ImageEncoder(nn.Module):
    """
    A ResNet-based encoder. Loads a standard ResNet-34 checkpoint
    with pretrained weights. Adjust layers and weights as desired
    for your actual use case.
    """
    def __init__(self, embed_dim=512, pretrained=True):
        super().__init__()
        self.model = models.resnet34(pretrained=pretrained)
        self.model.fc = nn.Identity()
        self.projection = (
            nn.Linear(512, embed_dim) if embed_dim != 512 else nn.Identity()
        )

    def forward(self, x):
        """
        x: [batch_size, 3, H, W] input images
        returns: [batch_size, embed_dim] image embeddings
        """
        features = self.model(x)              
        embeddings = self.projection(features) 
        return embeddings

class LinformerTextEncoder(nn.Module):
    """
    A text encoder that:
      1) Uses a Hugging Face tokenizer (HFTokenizer) externally for tokenization.
      2) Uses Linformer for self-attention (if installed), else falls back to an MLP.
    """
    def __init__(
        self,
        vocab_size=50257,     
        embed_dim=256,
        max_seq_len=64,
        n_heads=4,
        hidden_dim=512,
        n_layers=4,
        dropout=0.1
    ):
        """
        Args:
            vocab_size (int): Size of the tokenizer's vocabulary.
            embed_dim (int): Dimensionality for token embeddings.
            max_seq_len (int): Maximum input text length.
            n_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension in feedforward layers.
            n_layers (int): Number of Linformer/Transformer blocks.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        if Linformer is not None:
            self.encoder = Linformer(
                dim=embed_dim,
                seq_len=max_seq_len,
                depth=n_layers,
                heads=n_heads,
                k=64,         
                one_kv_head=True,
                share_kv=True
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim)
            )

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.LongTensor):
        """
        token_ids: [batch_size, seq_len] integer IDs for each token
        returns:   [batch_size, embed_dim] text embeddings
        """
        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}."
            )

        x = self.token_emb(token_ids)  
        positions = torch.arange(0, seq_len, device=token_ids.device).unsqueeze(0)
        x = x + self.pos_emb(positions) 

        x = self.dropout(x)

        if Linformer is not None:
            x = self.encoder(x) 
        else:
            # Fallback MLP
            x = x.view(-1, x.shape[-1])       
            x = self.encoder(x)               
            x = x.view(batch_size, seq_len, -1)

        text_emb = x.mean(dim=1)              

        text_emb = self.layer_norm(text_emb)
        return text_emb

class HFTextEncoder(nn.Module):
    """
    Wraps together a Hugging Face Tokenizer (from tokenizer.py) and the Linformer
    text encoder into a single module for convenience. 
    instead of manually calling the tokenizer in one place and the encoder in another.
    This is entirely optional.
    """
    def __init__(
        self,
        hf_tokenizer,
        text_encoder: LinformerTextEncoder
    ):
        """
        Args:
            hf_tokenizer: An instance of HFTokenizer.
            text_encoder: An instance of LinformerTextEncoder.
        """
        super().__init__()
        self.hf_tokenizer = hf_tokenizer
        self.text_encoder = text_encoder

    def forward(self, text_list):
        """
        text_list: a list of raw text strings, e.g. ["This is a cat", "Hello world!"]
        returns:   [batch_size, embed_dim] text embeddings
        """
        token_ids_batch = [self.hf_tokenizer(txt) for txt in text_list]
        max_len = max(len(t) for t in token_ids_batch)
        if max_len > self.text_encoder.max_seq_len:
            max_len = self.text_encoder.max_seq_len

        padded_ids = []
        for ids in token_ids_batch:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.hf_tokenizer.tokenizer.pad_token_id] * (max_len - len(ids))
            padded_ids.append(ids)

        token_ids_tensor = torch.LongTensor(padded_ids)  

        embeddings = self.text_encoder(token_ids_tensor)
        return embeddings
