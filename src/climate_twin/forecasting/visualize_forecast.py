from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class ForecastVisualizer:

    def __init__(self, cmap="Blues"):
        self.cmap = cmap

    def plot_day(
        self,
        forecast,
        day,
        save=False,
        output_dir="results/forecast"
    ):
        """
        Plot a single forecast day.
        """

        plt.figure(figsize=(6, 6))

        plt.imshow(
            forecast[day],
            cmap=self.cmap
        )

        plt.colorbar(label="Normalized Rainfall")

        plt.title(f"Forecast Day {day+1}")

        plt.axis("off")

        if save:

            output_dir = Path(output_dir)
            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            plt.savefig(
                output_dir / f"day_{day+1:03d}.png",
                dpi=200,
                bbox_inches="tight"
            )

        plt.show()

    def save_all(
        self,
        forecast,
        output_dir="results/forecast"
    ):

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for day in range(len(forecast)):

            plt.figure(figsize=(6,6))

            plt.imshow(
                forecast[day],
                cmap=self.cmap
            )

            plt.colorbar()

            plt.title(
                f"Forecast Day {day+1}"
            )

            plt.axis("off")

            plt.savefig(
                output_dir / f"day_{day+1:03d}.png",
                dpi=200,
                bbox_inches="tight"
            )

            plt.close()

        print(
            f"{len(forecast)} forecast maps saved."
        )

    def animate(
        self,
        forecast,
        interval=500,
        save=False,
        filename="results/forecast.gif"
    ):

        fig, ax = plt.subplots(figsize=(6,6))

        image = ax.imshow(
            forecast[0],
            cmap=self.cmap,
            animated=True
        )

        plt.colorbar(image)

        ax.axis("off")

        def update(frame):

            image.set_array(
                forecast[frame]
            )

            ax.set_title(
                f"Forecast Day {frame+1}"
            )

            return [image]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(forecast),
            interval=interval,
            blit=True
        )

        if save:

            filename = Path(filename)

            filename.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            ani.save(
                filename,
                writer="pillow"
            )

            print("Animation saved.")

        plt.show()