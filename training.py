
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm 

from flickr_datasets import Flickr30kDataset
from models.encoder import ImageEncoder, LinformerTextEncoder
from models.projection_MLP import ProjectionMLP
from models.dynamic_temperature import DynamicTemperature
from models.sam import SAM
from models.tokenizer import HFTokenizer

def clip_style_contrastive_loss(logits):
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def train_mini_clip(
    train_csv= 'flickr_annotations_30k.csv',
    train_root="flickr30k-images",
    val_csv=None,
    val_root=None,
    output_dir="./checkpoints",
    num_epochs=50,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-2,
    rho=0.05,
    device="cuda",
    save_every=10
):

    os.makedirs(output_dir, exist_ok=True)


    print("Loading datasets...")
    train_dataset = Flickr30kDataset(
        csv_path=train_csv,
        root_dir=train_root,
        split="train"  
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = None
    if val_csv and val_root:
        val_dataset = Flickr30kDataset(
            csv_path=val_csv,
            root_dir=val_root,
            split="val"  
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    print("Initializing model components...")

    image_encoder = ImageEncoder(embed_dim=256, pretrained=True).to(device)
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
    dynamic_temp = DynamicTemperature(hidden_dim=16).to(device)

    tokenizer = HFTokenizer(pretrained_name="gpt2", max_length=64)

    all_params = (
        list(image_encoder.parameters()) +
        list(text_encoder.parameters()) +
        list(projection_mlp.parameters()) +
        list(dynamic_temp.parameters())
    )

    optimizer = SAM(
        all_params,
        base_optimizer=torch.optim.AdamW,
        rho=rho,
        lr=lr,
        weight_decay=weight_decay
    )

    scaler = GradScaler()

    print("Starting training...")

    global_step = 0
    for epoch in range(num_epochs):
        image_encoder.train()
        text_encoder.train()
        projection_mlp.train()
        dynamic_temp.train()

        epoch_loss = 0.0
        start_time = time.time()

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, (images, raw_captions) in enumerate(train_iter):
            images = images.to(device, non_blocking=True)

            token_ids_batch = [tokenizer(caption) for caption in raw_captions]

            max_len = max(len(t) for t in token_ids_batch)
            if max_len > text_encoder.max_seq_len:
                max_len = text_encoder.max_seq_len

            padded_ids = []
            for t_ids in token_ids_batch:
                if len(t_ids) > max_len:
                    t_ids = t_ids[:max_len]
                else:
                    pad_id = tokenizer.tokenizer.pad_token_id
                    t_ids = t_ids + [pad_id]*(max_len - len(t_ids))
                padded_ids.append(t_ids)

            text_tokens = torch.LongTensor(padded_ids).to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                img_emb = image_encoder(images)           
                txt_emb = text_encoder(text_tokens)     
                img_proj = projection_mlp(img_emb)         
                txt_proj = projection_mlp(txt_emb)         

                img_proj = F.normalize(img_proj, dim=-1)
                txt_proj = F.normalize(txt_proj, dim=-1)

                sim_matrix = img_proj @ txt_proj.t()      
                temperature = dynamic_temp(sim_matrix)    
                logits = sim_matrix * temperature
                loss = clip_style_contrastive_loss(logits)

            scaler.scale(loss).backward()
            optimizer.first_step(scaler=scaler, zero_grad=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                img_emb2 = image_encoder(images)
                txt_emb2 = text_encoder(text_tokens)
                img_proj2 = projection_mlp(img_emb2)
                txt_proj2 = projection_mlp(txt_emb2)

                img_proj2 = F.normalize(img_proj2, dim=-1)
                txt_proj2 = F.normalize(txt_proj2, dim=-1)

                sim_matrix2 = img_proj2 @ txt_proj2.t()
                temperature2 = dynamic_temp(sim_matrix2)
                logits2 = sim_matrix2 * temperature2

                loss2 = clip_style_contrastive_loss(logits2)

            scaler.scale(loss2).backward()
            optimizer.second_step(scaler=scaler, zero_grad=True)
            scaler.update()

            batch_loss = 0.5 * (loss.item() + loss2.item())
            epoch_loss += batch_loss
            global_step += 1

            if (batch_idx % 50) == 0:
                current_lr = optimizer.base_optimizer.param_groups[0]["lr"]
                train_iter.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                    "lr": f"{current_lr:.6f}",
                    "temp": f"{temperature.item():.4f}"
                })

        epoch_loss /= len(train_loader)
        duration = time.time() - start_time

        print(f"\n==> Epoch {epoch+1} finished in {duration:.2f}s | "
              f"Avg Train Loss: {epoch_loss:.4f}")

        if val_loader is not None:
            image_encoder.eval()
            text_encoder.eval()
            projection_mlp.eval()
            dynamic_temp.eval()

            val_loss = 0.0
            val_iter = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
            with torch.no_grad():
                for images_val, raw_captions_val in val_iter:
                    images_val = images_val.to(device, non_blocking=True)

                    val_ids_batch = [tokenizer(txt) for txt in raw_captions_val]
                    max_len_val = max(len(t) for t in val_ids_batch)
                    if max_len_val > text_encoder.max_seq_len:
                        max_len_val = text_encoder.max_seq_len

                    padded_val = []
                    for t_ids_v in val_ids_batch:
                        if len(t_ids_v) > max_len_val:
                            t_ids_v = t_ids_v[:max_len_val]
                        else:
                            pad_id = tokenizer.tokenizer.pad_token_id
                            t_ids_v += [pad_id]*(max_len_val - len(t_ids_v))
                        padded_val.append(t_ids_v)

                    text_tokens_val = torch.LongTensor(padded_val).to(device)

                    with autocast():
                        img_val_emb = image_encoder(images_val)
                        txt_val_emb = text_encoder(text_tokens_val)
                        img_val_proj = projection_mlp(img_val_emb)
                        txt_val_proj = projection_mlp(txt_val_emb)
                        img_val_proj = F.normalize(img_val_proj, dim=-1)
                        txt_val_proj = F.normalize(txt_val_proj, dim=-1)

                        sim_val = img_val_proj @ txt_val_proj.t()
                        temp_val = dynamic_temp(sim_val)
                        logits_val = sim_val * temp_val

                        loss_val = clip_style_contrastive_loss(logits_val)
                    val_loss += loss_val.item()

            val_loss /= len(val_loader)
            print(f"Validation Loss after Epoch {epoch+1}: {val_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            ckpt_name = f"checkpoint_epoch_{epoch+1}.pt"
            ckpt_path = os.path.join(output_dir, ckpt_name)
            torch.save({
                "epoch": epoch + 1,
                "image_encoder": image_encoder.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "proj_mlp": projection_mlp.state_dict(),
                "dyn_temp": dynamic_temp.state_dict(),
                "optimizer": optimizer.state_dict(),

            }, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")


def main():
    train_mini_clip()


if __name__ == "__main__":
    main()
