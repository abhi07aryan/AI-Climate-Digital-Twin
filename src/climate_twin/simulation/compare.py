from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class ClimateComparison:
    """
    Compare two climate forecasts.

    Baseline Forecast
            vs
    Scenario Forecast
    """

    def __init__(self):
        pass

    def difference(self, baseline, scenario):
        """
        Difference = Scenario - Baseline
        """

        baseline = np.asarray(baseline)
        scenario = np.asarray(scenario)

        return scenario - baseline

    def statistics(self, baseline, scenario):

        diff = self.difference(
            baseline,
            scenario
        )

        stats = {

            "mean_change":
                float(np.nanmean(diff)),

            "max_increase":
                float(np.nanmax(diff)),

            "max_decrease":
                float(np.nanmin(diff)),

            "std_change":
                float(np.nanstd(diff))

        }

        return stats

    def plot_day(
        self,
        baseline,
        scenario,
        day=0,
        save=False,
        output="results/comparison"
    ):

        diff = self.difference(
            baseline,
            scenario
        )

        output = Path(output)

        output.mkdir(
            parents=True,
            exist_ok=True
        )

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(18,6)
        )

        vmin = min(
            np.nanmin(baseline[day]),
            np.nanmin(scenario[day])
        )

        vmax = max(
            np.nanmax(baseline[day]),
            np.nanmax(scenario[day])
        )

        im1 = ax[0].imshow(
            baseline[day],
            cmap="Blues",
            vmin=vmin,
            vmax=vmax
        )

        ax[0].set_title("Baseline")

        ax[0].axis("off")

        im2 = ax[1].imshow(
            scenario[day],
            cmap="Blues",
            vmin=vmin,
            vmax=vmax
        )

        ax[1].set_title("Scenario")

        ax[1].axis("off")

        im3 = ax[2].imshow(
            diff[day],
            cmap="RdBu_r"
        )

        ax[2].set_title("Difference")

        ax[2].axis("off")

        plt.colorbar(
            im1,
            ax=ax[:2],
            shrink=0.8
        )

        plt.colorbar(
            im3,
            ax=ax[2],
            shrink=0.8
        )

        plt.tight_layout()

        if save:

            plt.savefig(

                output /
                f"comparison_day_{day+1:03d}.png",

                dpi=200

            )

        plt.show()

    def print_statistics(
        self,
        baseline,
        scenario
    ):

        stats = self.statistics(
            baseline,
            scenario
        )

        print()

        print("="*45)

        print("Climate Scenario Comparison")

        print("="*45)

        print(
            f"Mean Change     : {stats['mean_change']:.4f}"
        )

        print(
            f"Maximum Increase: {stats['max_increase']:.4f}"
        )

        print(
            f"Maximum Decrease: {stats['max_decrease']:.4f}"
        )

        print(
            f"Std Deviation   : {stats['std_change']:.4f}"
        )

        print("="*45)