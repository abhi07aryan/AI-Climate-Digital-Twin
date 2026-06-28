import numpy as np


class ClimateMetrics:
    """
    Evaluation metrics for climate prediction.

    All metrics ignore NaN values automatically.
    """

    @staticmethod
    def _mask(y_true, y_pred):
        """
        Remove NaN pixels from both arrays.
        """

        mask = (
            ~np.isnan(y_true)
        ) & (
            ~np.isnan(y_pred)
        )

        return y_true[mask], y_pred[mask]

    @staticmethod
    def mse(y_true, y_pred):

        y_true, y_pred = ClimateMetrics._mask(
            y_true,
            y_pred
        )

        return np.mean(
            (y_true - y_pred) ** 2
        )

    @staticmethod
    def rmse(y_true, y_pred):

        return np.sqrt(
            ClimateMetrics.mse(
                y_true,
                y_pred
            )
        )

    @staticmethod
    def mae(y_true, y_pred):

        y_true, y_pred = ClimateMetrics._mask(
            y_true,
            y_pred
        )

        return np.mean(
            np.abs(
                y_true - y_pred
            )
        )

    @staticmethod
    def r2(y_true, y_pred):

        y_true, y_pred = ClimateMetrics._mask(
            y_true,
            y_pred
        )

        ss_res = np.sum(
            (y_true - y_pred) ** 2
        )

        ss_tot = np.sum(
            (y_true - np.mean(y_true)) ** 2
        )

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    @staticmethod
    def pearson(y_true, y_pred):

        y_true, y_pred = ClimateMetrics._mask(
            y_true,
            y_pred
        )

        if len(y_true) < 2:
            return 0.0

        return np.corrcoef(
            y_true,
            y_pred
        )[0, 1]