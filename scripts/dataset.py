import torch
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    DATASET_NAME = "lerobot/aloha_sim_insertion_human"
    dataset = LeRobotDataset(DATASET_NAME)
    
    print(f"Total samples: {len(dataset)}")
    sample = dataset[0]
    print("Sample keys and types:")
    for k, v in sample.items():
        print(f"  '{k}': type={type(v)}, shape={getattr(v, 'shape', None)}")
    
    # DataLoader batching
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    
    #Visualize a camera image
    img_keys = [k for k in batch if 'image' in k.lower() or 'camera' in k.lower()]
    img_key = None
    for k in img_keys:
        t = batch[k]
        if isinstance(t, torch.Tensor) and t.ndim in (4, 5):
            img_key = k
            break
    if img_key:
        images = batch[img_key]
        # For 5D: batch x time x C x H x W, use the last time index if so
        if images.ndim == 5:
            images = images[:, -1]  # (batch, C, H, W)
        plt.figure(figsize=(12,3))
        for i in range(images.shape[0]):
            img = images[i]
            if img.shape[0] in (1, 3):  # CHW
                img_disp = img.permute(1,2,0).detach().cpu().numpy()
                if img_disp.dtype != "uint8":
                    img_disp = (img_disp * 255).clip(0,255).astype("uint8")
                plt.subplot(1, images.shape[0], i+1)
                plt.imshow(img_disp)
                plt.title(f"Sample {i}")
                plt.axis('off')
        plt.suptitle(f"Images from batch ('{img_key}')")
        plt.tight_layout()
        plt.show()
    else:
        print("No image/camera field found for visualization in the batch.")
        print("Batch keys:", list(batch.keys()))

if __name__ == "__main__":
    main()