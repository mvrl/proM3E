import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class ProM3EDataset(Dataset):
    """
    Multimodal Dataset for ProM3E.
    
    This dataset handles loading and sampling from two different feature sets:
    1. Taxabind: Comprehensive set with all 6 modalities unmasked (including audio).
    2. iNat: Set where modality 5 (audio) is typically unavailable/zeroed.
    
    The dataset is designed to return batches of samples to simplify the training loop 
    when dealing with datasets of varying modality presence.
    
    Modalities:
    [0: Image, 1: Sat, 2: Loc, 3: Env, 4: Text, 5: Audio]
    """
    def __init__(
        self, 
        taxabind_path: str, 
        inat_path: str, 
        batch_size: int = 1024, 
        split: str = 'train',
        inat_split_size: int = 80000
    ):
        super().__init__()
        
        # Load raw data
        try:
            taxabind_data = np.load(taxabind_path, allow_pickle=True)
            inat_data = np.load(inat_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading datasets: {e}")
            raise

        # Helper to extract and stack from dict-like archive
        def stack_modalities(data, has_audio=True):
            data_dict = data[()] if isinstance(data, np.ndarray) and data.dtype == object else data
            
            image = np.array(data_dict['image'])
            sat = np.array(data_dict['sat'])
            loc = np.array(data_dict['loc'])
            env = np.array(data_dict['env'])
            text = np.array(data_dict['text'])
            
            if has_audio:
                audio = np.array(data_dict['sound'])
            else:
                audio = np.zeros_like(image)
                
            return np.stack((image, sat, loc, env, text, audio), axis=1)

        self.taxabind_embeds = stack_modalities(taxabind_data, has_audio=True)
        self.inat_embeds = stack_modalities(inat_data, has_audio=False)
        
        # Train/Val Split for iNat (Taxabind is typically smaller/pre-defined)
        if split == 'train':
            self.inat_embeds = self.inat_embeds[:inat_split_size]
        else:
            self.inat_embeds = self.inat_embeds[inat_split_size:]
            
        self.batch_size = batch_size
        print(f"ProM3E Dataset initialized (split: {split})")
        print(f"  - Taxabind samples: {len(self.taxabind_embeds)}")
        print(f"  - iNat samples:     {len(self.inat_embeds)}")

    def __len__(self) -> int:
        # Total number of batches per epoch
        return (len(self.taxabind_embeds) + len(self.inat_embeds)) // self.batch_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Samples a full batch of modalities.
        Returns:
            - batch_tensor: torch.Tensor [batch_size, 6, embed_dim]
            - audio_flag: int (1 if Taxabind, 0 if iNat)
        """
        # Alternate between the two datasets randomly
        if torch.rand(1) < 0.5:
            indices = np.random.choice(len(self.taxabind_embeds), self.batch_size, replace=False)
            batch = self.taxabind_embeds[indices]
            audio_flag = 1
        else:
            indices = np.random.choice(len(self.inat_embeds), self.batch_size, replace=False)
            batch = self.inat_embeds[indices]
            audio_flag = 0
            
        # Float32 and normalized
        batch_tensor = torch.from_numpy(batch).float()
        batch_tensor = torch.nn.functional.normalize(batch_tensor, dim=-1)
        
        return batch_tensor, audio_flag

class ProM3EInferenceDataset(Dataset):
    """Generic dataset for running inference over a set of embeddings."""
    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        data_dict = data[()]
        self.embeds = np.stack((
            np.array(data_dict['image']),
            np.array(data_dict['sat']),
            np.array(data_dict['loc']),
            np.array(data_dict['env']),
            np.array(data_dict['text']),
            np.array(data_dict['sound'])
        ), axis=1)
        
    def __len__(self):
        return len(self.embeds)
        
    def __getitem__(self, idx):
        # Single item, normalized
        tensor = torch.from_numpy(self.embeds[idx]).float()
        return torch.nn.functional.normalize(tensor, dim=-1)

if __name__ == "__main__":
    # Test script for local verification
    print("Testing ProM3E Dataset...")
    # Requires dummy files to be present, otherwise will catch error
    try:
        ds = ProM3EDataset('dummy_taxabind.npy', 'dummy_inat.npy', batch_size=32)
        print(f"Successfully created dataset of length {len(ds)}")
    except:
        print("Dataset initialization skipped (files not found in local path).")
