import torch



def one_hot(batch_size, num_classes, labels):
    # labels = labels.to(torch.device('cpu'))
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    x = torch.zeros(batch_size, num_classes)
    x.scatter_(1, labels.long(), 1)
    return x



