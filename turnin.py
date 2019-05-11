# 1: Preprocessing

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 2: Model

class Net(nn.Module):
    def __init__(self, threshold):
        super(Net, self).__init__()
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 60000, 28)

    def forward(self, x):
        x = torch.abs(self.conv(x))
        x[x > self.threshold] = 0
        x = x.nonzero()
        if (x.shape[0] == 1):
            return torch.tensor(0)
        return torch.tensor(1)

# 3: Postprocess

# Post Processing is included in the model itself

# 4: Written explanation

# final_project.report.pdf