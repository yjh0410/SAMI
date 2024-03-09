import torch
from thop import profile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def FLOPs_and_Params(model, size):
    x = torch.randn(1, 3, size, size).to(device)
    model.eval()

    flops, params = profile(model, inputs=(x, ))
    print('GFLOPs : ', flops / 1e9 * 2)
    print('Params : ', params / 1e6, ' M')

    model.train()


if __name__ == "__main__":
    pass
