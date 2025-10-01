import argparse, torch, torch.nn as nn
from train import SmallCNN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None, help='(optional) path to state_dict')
    ap.add_argument('--out', default='model.onnx')
    args = ap.parse_args()

    model = SmallCNN()
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(sd)
    model.eval()

    dummy = torch.randn(1,1,28,28)
    torch.onnx.export(model, dummy, args.out, input_names=['input'], output_names=['logits'], dynamic_axes={'input':{0:'batch'},'logits':{0:'batch'}}, opset_version=13)
    print('exported to', args.out)

if __name__ == '__main__':
    main()
