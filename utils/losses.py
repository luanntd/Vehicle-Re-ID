import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """Triplet Loss for Re-ID training"""
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, features, labels):
        """
        Args:
            features: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size)
        """
        n = features.size(0)
        
        # Compute pairwise distance matrix
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(features, features.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        
        # For each anchor, find the hardest positive and negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        for i in range(n):
            # Hardest positive
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            # Hardest negative
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss

class CenterLoss(nn.Module):
    """Center Loss for Re-ID training"""
    
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu and torch.cuda.is_available()  # Only use GPU if available

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size)
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss