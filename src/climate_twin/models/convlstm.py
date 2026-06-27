import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size=3
    ):

        super().__init__()

        padding = kernel_size // 2

        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):

        combined = torch.cat([x, h], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g

        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
class ConvLSTM(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels=1
    ):

        super().__init__()

        self.cell = ConvLSTMCell(
            input_channels,
            hidden_channels
        )

        self.output = nn.Conv2d(
            hidden_channels,
            output_channels,
            kernel_size=1
        )

    def forward(self, x):

        batch, seq, channels, H, W = x.shape

        device = x.device

        h = torch.zeros(
            batch,
            self.cell.hidden_channels,
            H,
            W,
            device=device
        )

        c = torch.zeros_like(h)

        for t in range(seq):

            h, c = self.cell(
                x[:, t],
                h,
                c
            )

        return self.output(h)