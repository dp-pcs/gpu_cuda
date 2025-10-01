import argparse, time, csv, os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x): return self.net(x)

def get_device(name):
    name = name.lower()
    if name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if name == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu', choices=['cpu','cuda','mps'])
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device} (cuda avail={torch.cuda.is_available()})")

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = SmallCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    total_start = time.time()
    step_times = []

    model.train()
    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            step_times.append(time.time() - t0)
            if (i+1) % 100 == 0:
                print(f"epoch {epoch+1} step {i+1} loss {loss.item():.4f}")
            # keep demo fast
            if i > 200: break

    total_time = time.time() - total_start
    ex_per_sec = (len(step_times) * args.batch_size) / total_time
    p95 = sorted(step_times)[int(0.95*len(step_times))-1]

    os.makedirs('results', exist_ok=True)
    out_csv = 'results/metrics.csv'
    new_file = not os.path.exists(out_csv)
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if new_file: w.writerow(['device','batch_size','epochs','total_time_sec','examples_per_sec','p95_step_sec'])
        w.writerow([str(device), args.batch_size, args.epochs, round(total_time,3), round(ex_per_sec,2), round(p95,4)])

    print(f"Total time: {total_time:.2f}s | ex/s: {ex_per_sec:.2f} | p95 step: {p95:.4f}s")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('CUDA memory allocated (MB):', torch.cuda.memory_allocated()/1e6)

if __name__ == '__main__':
    main()
