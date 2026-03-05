"""
Renderer for MetaLore Simulator.

Provides visualization using matplotlib and pygame.
"""

import string
import numpy as np
import pygame
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from metalore.visualization.symbols import BS_SYMBOL, SENSOR_SYMBOL


class Renderer:
    """Renders the simulation environment using matplotlib and pygame."""

    def __init__(self, lower: float, upper: float):
        self.closed = False

        self.window = None
        self.clock = None

        # Cached isolines (computed once on first render)
        self.conn_isolines = None
        self.mb_isolines = None

        # Cached colormap and normalizer
        self.colormap = cm.get_cmap("RdYlGn")
        self.unorm = plt.Normalize(lower, upper)


    def render(self, env, mode: str = "human"):
        """Render the environment."""
        if self.closed:
            return None

        # Set up matplotlib figure & axis configuration
        dpi = plt.rcParams['figure.dpi']
        fx = max(3.0 / 2.0 * 1.25 * env.width / dpi, 8.0)
        fy = max(1.25 * env.height / dpi, 5.0)
        fig = plt.figure(figsize=(fx, fy))
        gs = fig.add_gridspec(
            ncols=2,
            nrows=3,
            width_ratios=(4, 2),
            height_ratios=(2, 3, 3),
            hspace=0.45,
            wspace=0.2,
            top=0.95,
            bottom=0.15,
            left=0.025,
            right=0.955,
        )

        sim_ax = fig.add_subplot(gs[:, 0])
        metrics_ax = fig.add_subplot(gs[0, 1])
        bw_alloc_ax = fig.add_subplot(gs[1, 1])
        comp_alloc_ax = fig.add_subplot(gs[2, 1])

        # Render each component
        self.render_simulation(env, sim_ax)
        self.render_metrics(env, metrics_ax)
        self.render_bw_allocation(env, bw_alloc_ax)
        self.render_comp_allocation(env, comp_alloc_ax)

        # Convert to image
        fig.align_ylabels((bw_alloc_ax, comp_alloc_ax))
        window_size = tuple(map(int, fig.get_size_inches() * fig.dpi))
        canvas = FigureCanvas(fig)
        canvas.draw()
        plt.close(fig)

        if mode == "rgb_array" or mode is None:
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            return data.reshape(canvas.get_width_height()[::-1] + (3,))

        elif mode == "human":
            data = canvas.buffer_rgba()
            size = canvas.get_width_height()

            # Set up pygame window
            if self.window is None:
                pygame.init()
                self.clock = pygame.time.Clock()

                self.window = pygame.display.set_mode(window_size)
                pygame.display.set_icon(pygame.Surface((0, 0)))
                pygame.display.set_caption("MetaLore Environment")

            # Clear and draw
            self.window.fill("white")
            screen = pygame.display.get_surface()
            plot = pygame.image.frombuffer(data, size, "RGBA")
            screen.blit(plot, (0, 0))
            pygame.display.flip()

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            return None

        else:
            raise ValueError(f"Invalid rendering mode: {mode}")

    def render_simulation(self, env, ax: plt.Axes) -> None:
        """Render the simulation view with entities."""

        colormap = self.colormap
        unorm = self.unorm

        # Plot inactive UEs as grayed out
        active_ue_set = set(env.utilities_ue.keys())
        for ue in env.users.values():
            if ue not in active_ue_set:
                ax.scatter(
                    ue.x, ue.y,
                    s=200, zorder=1,
                    color="lightgray", marker="o", alpha=0.5,
                )
                ax.annotate(ue.id, xy=(ue.x, ue.y), ha="center", va="center",
                            color="gray", alpha=0.5)

        # Plot active UEs colored by utility
        for ue, utility in env.utilities_ue.items():
            utility = env.utility.unscale(utility)
            color = colormap(unorm(utility))

            ax.scatter(
                ue.x,
                ue.y,
                s=200,
                zorder=2,
                color=color,
                marker="o",
            )
            ax.annotate(ue.id, xy=(ue.x, ue.y), ha="center", va="center")

        for bs in env.stations.values():
            ax.plot(
                bs.x, bs.y,
                marker=BS_SYMBOL,
                markersize=30,
                markeredgewidth=0.1,
                color="black",
            )
            bs_id = string.ascii_uppercase[bs.id]
            ax.annotate(
                bs_id,
                xy=(bs.x, bs.y),
                xytext=(0, -25),
                ha="center",
                va="bottom",
                textcoords="offset points",
            )

            # Plot BS coverage ranges
            #ax.scatter(*self.conn_isolines[bs], color="gray", s=3)
            #ax.scatter(*self.mb_isolines[bs], color="black", s=3)

        for bs in env.stations.values():
            for ue in env.connections_ue[bs]:
                if ue not in env.utilities_ue:
                    continue
                share = env.utility.unscale(env.utilities_ue[ue])
                color = colormap(unorm(share))

                # add black background/borders for lines for visibility
                ax.plot(
                    [ue.x, bs.x],
                    [ue.y, bs.y],
                    color=color,
                    path_effects=[
                        pe.SimpleLineShadow(shadow_color="black"),
                        pe.Normal(),
                    ],
                    linewidth=3,
                    zorder=-1,
                )

        for sensor in env.sensors.values():
            ax.plot(
                sensor.x, sensor.y,
                marker=SENSOR_SYMBOL,
                markersize=10,
                markeredgewidth=0.1,
                color="blue",
            )
            sensor_id = string.ascii_uppercase[sensor.id]
            ax.annotate(
                sensor_id,
                xy=(sensor.x, sensor.y),
                xytext=(0, -15),
                ha="center",
                va="bottom",
                textcoords="offset points",
                fontsize="8",
            )

        # Show border, hide ticks
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xlim([0, env.width])
        ax.set_ylim([0, env.height])

    def render_metrics(self, env, ax: plt.Axes) -> None:
        """Render the info dashboard."""
        aori_series = [v for v in env.metrics.jobs["mean_aori"] if v is not None]
        aosi_series = [v for v in env.metrics.jobs["mean_aosi"] if v is not None]

        cur_aori = f"{aori_series[-1]:.2f}" if aori_series else "—"
        cur_aosi = f"{aosi_series[-1]:.2f}" if aosi_series else "—"
        avg_aori = f"{sum(aori_series)/len(aori_series):.2f}" if aori_series else "—"
        avg_aosi = f"{sum(aosi_series)/len(aosi_series):.2f}" if aosi_series else "—"

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        rows = ["Current", "Average"]
        cols = ["AoRI", "AoSI"]
        text = [
            [cur_aori, cur_aosi],
            [avg_aori, avg_aosi],
        ]

        table = ax.table(
            text,
            rowLabels=rows,
            colLabels=cols,
            cellLoc="center",
            edges="B",
            loc="upper center",
            bbox=[0.0, -0.25, 1.0, 1.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)


    def render_bw_allocation(self, env, ax: plt.Axes) -> None:
        bw_splits = env.metrics.actions['bw_split']
        time = np.arange(len(bw_splits))
        ax.plot(time, bw_splits, linewidth=1, color="blue", label="UE")
        ax.plot(time, 1 - np.array(bw_splits), linewidth=1, color="green", label="Sensor")

        ax.set_xlabel("Time")
        ax.set_ylabel("BW Allocation")
        ax.set_xlim([0.0, env.EP_MAX_TIME])
        ax.set_ylim([0.0, 1.0])
        ax.legend(loc="upper right", fontsize=8)

    def render_comp_allocation(self, env, ax: plt.Axes) -> None:
        comp_splits = env.metrics.actions['comp_split']
        time = np.arange(len(comp_splits))
        ax.plot(time, comp_splits, linewidth=1, color="blue", label="UE")
        ax.plot(time, 1 - np.array(comp_splits), linewidth=1, color="green", label="Sensor")

        ax.set_xlabel("Time")
        ax.set_ylabel("Comp. Allocation")
        ax.set_xlim([0.0, env.EP_MAX_TIME])
        ax.set_ylim([0.0, 1.0])
        ax.legend(loc="upper right", fontsize=8)

    def close(self) -> None:
        """Close the renderer and release resources."""
        pygame.quit()
        self.window = None
        self.closed = True
