import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units=10):
        super(CNN, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2,2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*5*5, out_features=output_shape),
        )

    def forward(self, x) ->  torch.Tensor:
        x = self.conv_block_1(x)
        #print(f"shape after block 1: {x.shape}")
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = CNN(input_shape=1, output_shape=3)
    dummy_tensor = torch.rand(1,1,5,5)
    print(f"dummy input: {dummy_tensor}")
    y_logit = model(dummy_tensor)
    print(f"logit: {y_logit}")
    print(f"shape logit: {y_logit.shape}")
