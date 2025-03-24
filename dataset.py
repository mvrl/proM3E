import numpy as np
from torch.utils.data import Dataset
import torch

class M3EDataset(Dataset):
    def __init__(self, data_path, data_path_inat, batch_size=1024, split='train'):
        self.data = np.load(data_path, allow_pickle=True)
        self.data_inat = np.load(data_path_inat, allow_pickle=True)
        self.embeds_taxabind = np.stack((np.array(self.data[()]['image']),
                                np.array(self.data[()]['sat']),
                                np.array(self.data[()]['loc']),
                                np.array(self.data[()]['env']),
                                np.array(self.data[()]['text']),
                                np.array(self.data[()]['sound'])), axis=1)

        self.embeds_inat = np.stack((np.array(self.data_inat[()]['image']),
                                np.array(self.data_inat[()]['sat']),
                                np.array(self.data_inat[()]['loc']),
                                np.array(self.data_inat[()]['env']),
                                np.array(self.data_inat[()]['text']),
                                np.zeros(np.array(self.data_inat[()]['image']).shape)), axis=1)
        if split=='train':
            self.embeds_inat = self.embeds_inat[:80000]
        else:
            self.embeds_inat = self.embeds_inat[80000:]
        
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.embeds_taxabind)+len(self.embeds_inat))//self.batch_size

    def __getitem__(self, idx):
        if torch.rand(1) < 0.5:
            list_ids = np.random.choice(len(self.embeds_taxabind), self.batch_size, replace=False)
            return torch.nn.functional.normalize(torch.from_numpy(self.embeds_taxabind[list_ids]), dim=-1), 1
        else:
            list_ids = np.random.choice(len(self.embeds_inat), self.batch_size, replace=False)
            return torch.nn.functional.normalize(torch.from_numpy(self.embeds_inat[list_ids]), dim=-1), 0

class M3EDatasetV1(Dataset):
    def __init__(self, data_path, data_path_inat, batch_size=1024, split='train'):
        self.data = np.load(data_path, allow_pickle=True)
        self.data_inat = np.load(data_path_inat, allow_pickle=True)
        self.embeds_taxabind = np.stack((np.array(self.data[()]['image']),
                                np.array(self.data[()]['sat']),
                                np.array(self.data[()]['loc']),
                                np.array(self.data[()]['env']),
                                np.array(self.data[()]['text']),
                                np.array(self.data[()]['sound'])), axis=1)

        self.embeds_inat = np.stack((np.array(self.data_inat[()]['image']),
                                np.array(self.data_inat[()]['sat']),
                                np.array(self.data_inat[()]['loc']),
                                np.array(self.data_inat[()]['env']),
                                np.array(self.data_inat[()]['text']),
                                np.zeros(np.array(self.data_inat[()]['image']).shape)), axis=1)
        if split=='train':
            self.embeds_inat = self.embeds_inat[:80000]
        else:
            self.embeds_inat = self.embeds_inat[80000:]
        
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.embeds_taxabind)+len(self.embeds_inat))//self.batch_size

    def __getitem__(self, idx):
        list_ids = np.random.choice(len(self.embeds_taxabind), self.batch_size, replace=False)
        return torch.nn.functional.normalize(torch.from_numpy(self.embeds_taxabind[list_ids]), dim=-1), 1

class M3EDatasetInference(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
        self.embeds_taxabind = np.stack((np.array(self.data[()]['image']),
                                np.array(self.data[()]['sat']),
                                np.array(self.data[()]['loc']),
                                np.array(self.data[()]['env']),
                                np.array(self.data[()]['text']),
                                np.array(self.data[()]['sound'])), axis=1)
    def __len__(self):
        return len(self.embeds_taxabind)
    def __getitem__(self, idx):
        return torch.nn.functional.normalize(torch.from_numpy(self.embeds_taxabind[idx]), dim=-1)

if __name__=='__main__':
    dataset = M3EDataset('embeds/embeds_test.npy', 'embeds/embeds_inat.npy', batch_size=1024, split='test')
    print(len(dataset))