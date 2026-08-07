import numpy as np
import torch


class RecursiveForecaster:
    """
    Recursive multi-step rainfall forecasting.

    Changes vs. the previous version
    --------------------------------
    1. forecast() no longer calls model.eval() unconditionally. That call
       silently disabled dropout, which made every Monte Carlo sample
       identical and every uncertainty estimate zero. Pass mc_dropout=True
       to keep dropout active while leaving BatchNorm in eval mode.

    2. feedback_alpha defaults to 1.0 (true recursion). The previous
       hard-coded 0.7 blended 30% of the previous day's rainfall into each
       fed-back step, which damps the forecast toward persistence and
       suppresses ensemble divergence.

    3. The rainfall channel index is a parameter rather than a hard-coded 0.
    """

    def __init__(
        self,
        model,
        device,
        rainfall_index=0,
        feedback_alpha=1.0,
        mc_dropout=False,
    ):

        self.model = model
        self.device = device
        self.rainfall_index = int(rainfall_index)
        self.feedback_alpha = float(feedback_alpha)
        self.mc_dropout = bool(mc_dropout)

    # ------------------------------------------------------------------
    # Inference mode
    # ------------------------------------------------------------------

    def _set_inference_mode(self):
        """
        Deterministic inference puts the whole model in eval mode.

        Monte Carlo Dropout needs dropout layers active but everything else
        in eval mode. Calling model.train() would also switch BatchNorm to
        batch statistics and update its running estimates on every forward
        pass, corrupting the checkpoint over repeated simulations.
        """

        self.model.eval()

        if not self.mc_dropout:
            return

        found = False

        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.train()
                found = True

        return found

    def has_dropout(self):
        return any(
            isinstance(m, torch.nn.modules.dropout._DropoutNd)
            for m in self.model.modules()
        )

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def forecast(self, initial_sequence, days):
        """
        Parameters
        ----------
        initial_sequence : ndarray, shape (window, channels, height, width)
        days : int, number of future days

        Returns
        -------
        list of predicted rainfall maps, each (height, width), in the same
        normalised units as the rainfall input channel.

        Notes
        -----
        Only the rainfall channel is updated between steps. All other
        channels are carried forward frozen from the last observed frame,
        so a temperature perturbation applied to the input sequence
        persists across the whole horizon, but seasonal evolution of
        temperature does not.
        """

        sequence = np.array(initial_sequence, dtype=np.float32, copy=True)

        if sequence.ndim != 4:
            raise ValueError(
                "Expected (window, channels, height, width), "
                f"received {sequence.shape}."
            )

        height, width = sequence.shape[2], sequence.shape[3]

        self._set_inference_mode()

        predictions = []

        alpha = self.feedback_alpha

        with torch.no_grad():

            for _ in range(days):

                x = torch.from_numpy(
                    np.ascontiguousarray(sequence)
                ).unsqueeze(0).to(self.device)

                pred = self.model(x)

                # Reshape rather than squeeze: squeeze() would also collapse
                # a spatial axis of length 1.
                pred = pred.reshape(height, width).cpu().numpy()

                predictions.append(pred)

                new_step = sequence[-1].copy()

                previous = sequence[-1][self.rainfall_index]

                new_step[self.rainfall_index] = (
                    alpha * pred + (1.0 - alpha) * previous
                )

                sequence = np.concatenate(
                    [sequence[1:], new_step[np.newaxis]],
                    axis=0,
                )

        return predictions