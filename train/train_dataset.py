from torch.utils.data import Dataset
import random
import math
import torchvision

class SiameseDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset, train, poses, dist_thresh):
        self.dataset = dataset
        self.train = train
        self.poses = poses
        positive_pairs = []
        negative_pairs = []

        for i in range(len(self.dataset)):
            for j in range(len(self.dataset)):
                if i == j: continue
                else:
                    if math.pow((poses[i][1] - poses[j][1]), 2) + math.pow((poses[i][2] - poses[j][2]), 2) < dist_thresh * dist_thresh:
                        positive_pairs.append([i,j,1])
                    else: 
                        negative_pairs.append([i,j,0])    

        random.shuffle(positive_pairs)
        random.shuffle(negative_pairs)
        negative_pairs = negative_pairs[:len(positive_pairs)]
        
        all_pairs = positive_pairs + negative_pairs
        split_idx = round(len(all_pairs) * 0.7) 
        self.train_pairs = all_pairs[:split_idx]
        self.test_pairs = all_pairs[split_idx+1:]

    def __getitem__(self, index):
        if self.train:
            img1, img2, target = self.train_pairs[index]
        else:
            img1, img2, target = self.test_pairs[index]
        img1 = self.dataset[img1]
        img2 = self.dataset[img2]
        ## normalize data 
        img1 /= img1.max()
        img2 /= img1.max()
        return (img1, img2), target

    def __len__(self):
        if self.train:
            return len(self.train_pairs)
        else:
            return len(self.test_pairs)