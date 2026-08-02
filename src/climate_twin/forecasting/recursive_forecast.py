import numpy as np
import torch


class RecursiveForecaster:

    def __init__(
        self,
        model,
        device
    ):

        self.model = model
        self.device = device

    def forecast(
        self,
        initial_sequence,
        days
    ):
        """
        Parameters
        ----------
        initial_sequence

            shape

            (window,
             channels,
             height,
             width)

        days

            number of future days

        Returns
        -------

        list of predicted rainfall maps
        """

        sequence = initial_sequence.copy()

        predictions = []

        self.model.eval()

        with torch.no_grad():

            for _ in range(days):

                x = torch.tensor(
                    sequence,
                    dtype=torch.float32
                )

                x = x.unsqueeze(0)

                x = x.to(self.device)

                pred = self.model(x)

                pred = pred.squeeze()

                pred = pred.cpu().numpy()

                predictions.append(pred)

                # -----------------------------
                # Update sequence
                # -----------------------------

                alpha = 0.7

                new_step = sequence[-1].copy()

                new_step[0] = (alpha * pred + (1-alpha) * sequence[-1][0])

                sequence = np.concatenate(

                    [
                        sequence[1:],
                        new_step[np.newaxis]
                    ],

                    axis=0

                )

        return predictions