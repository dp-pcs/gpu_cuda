import argparse, torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1,16,3,padding=1)
        self.c2 = nn.Conv2d(16,32,3,padding=1)
        self.fc = nn.Linear(32*7*7,10)
    def forward(self,x):
        x = torch.relu(self.c1(x)); x = torch.max_pool2d(x,2)
        x = torch.relu(self.c2(x)); x = torch.max_pool2d(x,2)
        x = torch.flatten(x,1); return self.fc(x)

def get_device(name):
    if name=='cuda' and torch.cuda.is_available(): return torch.device('cuda')
    if name=='mps' and hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu')
    args = ap.parse_args(); device = get_device(args.device)
    print('profiling on', device)
    ds = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    model = Small().to(device); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x,y = next(iter(dl)); x,y = x.to(device), y.to(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type=='cuda' else [ProfilerActivity.CPU], record_shapes=True) as prof:
        for _ in range(10):
            with record_function("step"):
                opt.zero_grad(set_to_none=True)
                out = model(x); loss = torch.nn.functional.cross_entropy(out,y)
                loss.backward(); opt.step()

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

if __name__ == '__main__':
    main()
