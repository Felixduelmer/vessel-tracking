import torch
from torch.functional import Tensor
from torch.nn.modules import conv


def main():
    convGruInputs = torch.tensor([[1, 2], [3, 4]])

    print(convGruInputs[1, :])


if __name__ == "__main__":

    main()
