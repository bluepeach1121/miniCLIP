import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from flickr_datasets import Flickr30kDataset
from models.encoder import ImageEncoder, LinformerTextEncoder
from models.projection_MLP import ProjectionMLP
from models.tokenizer import HFTokenizer


def evaluate_recall5(
    csv_path="flickr_annotations_30k.csv",
    images_root="flickr30k-images",
    split="test",       
    checkpoint_path="./checkpoints/checkpoint_epoch_50.pt",
    fixed_temp=0.07,    
    batch_size=32,
    device="cuda"
):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    image_encoder = ImageEncoder(embed_dim=256, pretrained=False).to(device)
    text_encoder = LinformerTextEncoder(
        vocab_size=50257,
        embed_dim=256,
        max_seq_len=64,
        n_heads=4,
        hidden_dim=512,
        n_layers=4,
        dropout=0.1
    ).to(device)
    projection_mlp = ProjectionMLP(embed_dim=256).to(device)

    image_encoder.load_state_dict(checkpoint["image_encoder"])
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    projection_mlp.load_state_dict(checkpoint["proj_mlp"])

    image_encoder.eval()
    text_encoder.eval()
    projection_mlp.eval()

    dataset = Flickr30kDataset(
        csv_path=csv_path,
        root_dir=images_root,
        split=split
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    tokenizer = HFTokenizer(pretrained_name="gpt2", max_length=64)

    all_image_embs = []
    all_text_embs = []
    all_indices = [] 


    print(f"Computing embeddings for split='{split}' with {len(dataset)} samples...")
    with torch.no_grad():
        for batch_idx, (images, raw_captions) in enumerate(tqdm(dataloader)):
            images = images.to(device, non_blocking=True)
            token_ids_batch = []
            for cap in raw_captions:
                token_ids = tokenizer(cap)
                if len(token_ids) > text_encoder.max_seq_len:
                    token_ids = token_ids[:text_encoder.max_seq_len]
                else:
                    pad_id = tokenizer.tokenizer.pad_token_id
                    token_ids = token_ids + [pad_id]*(text_encoder.max_seq_len - len(token_ids))
                token_ids_batch.append(token_ids)

            text_tokens = torch.LongTensor(token_ids_batch).to(device, non_blocking=True)

            img_emb = image_encoder(images)                
            txt_emb = text_encoder(text_tokens)             

            img_proj = projection_mlp(img_emb)              
            txt_proj = projection_mlp(txt_emb)             

            img_proj = F.normalize(img_proj, dim=-1)        
            txt_proj = F.normalize(txt_proj, dim=-1)        

            all_image_embs.append(img_proj.cpu())
            all_text_embs.append(txt_proj.cpu())

            all_indices.extend(range(batch_idx*batch_size, batch_idx*batch_size + images.size(0)))

    all_image_embs = torch.cat(all_image_embs, dim=0)  
    all_text_embs = torch.cat(all_text_embs, dim=0)    
    N = all_image_embs.size(0)
    print(f"Total samples embedded: {N}")

    print(f"Computing NxN similarity matrix with fixed temperature {fixed_temp}")
    sim_matrix = all_image_embs @ all_text_embs.t()  
    sim_matrix *= fixed_temp

    correct_at_5 = 0

    print("Evaluating Recall@5 (image-to-text).")

    # We'll use topk approach on each row
    # If i is in the top-5 of row i, that's a "hit" for R@5
    with torch.no_grad():
        for i in range(N):
            row = sim_matrix[i]          
            # topk indices
            _, topk_indices = torch.topk(row, k=5)  
            # if the correct text is 'i', check if i in topk_indices
            if i in topk_indices:
                correct_at_5 += 1

    recall_at_5 = 100.0 * correct_at_5 / N
    print(f"Recall@5: {recall_at_5:.2f}% on {split} split for N={N} samples.")


def main():
    evaluate_recall5(
        csv_path="flickr_annotations_30k.csv",
        images_root="flickr30k-images",
        split="test",
        checkpoint_path="./checkpoints/checkpoint_epoch_50.pt",
        fixed_temp=0.07,   
        batch_size=32,
        device="cuda"
    )

if __name__ == "__main__":
    main()
