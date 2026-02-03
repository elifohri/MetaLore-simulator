"""
Renderer for MetaLore Simulator.

Provides visualization using matplotlib and pygame.
"""

import string
import numpy as np
import pygame
from pygame import Surface
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from metalore.visualization.symbols import BS_SYMBOL, UE_SYMBOL, SENSOR_SYMBOL
from metalore.visualization.utilities import BoundedLogUtility

class Renderer:
    

    def __init__(self, env):

        self.env = env
        self.closed = False

        self.window = None
        self.clock = None


        self.bs_symbol = BS_SYMBOL
        self.ue_symbol = UE_SYMBOL
        self.sensor_symbol = SENSOR_SYMBOL



    def render(self, mode: str = "human"):
        """Render the environment."""
        if self.closed:
            return None

        # set up matplotlib figure & axis configuration
        fig = plt.figure()
        fx = max(3.0 / 2.0 * 1.25 * self.env.width / fig.dpi, 8.0)
        fy = max(1.25 * self.env.height / fig.dpi, 5.0)
        plt.close()
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

        sim_ax = fig.add_subplot(gs[:, 0])          # Main simulation environment
        dash_ax = fig.add_subplot(gs[0, 1])         # Info dahsboard
        metrics_ax = fig.add_subplot(gs[1, 1])      # Metrics
        conn_ax = fig.add_subplot(gs[2, 1])         # Connections

        # Render each component
        self.render_simulation(sim_ax)
        #self.render_dashboard(dash_ax)
        #self.render_metrics(metrics_ax)
        #self.render_connections(conn_ax)

        # Convert to image
        fig.align_ylabels((metrics_ax, conn_ax))
        canvas = FigureCanvas(fig)
        canvas.draw()
        plt.close(fig)

        if mode == "rgb_array" or mode is None:
            # Return RGB array
            data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            return data.reshape(canvas.get_width_height()[::-1] + (3,))
    
        elif mode == "human":
            # Render on pygame surface
            data = canvas.buffer_rgba()
            size = canvas.get_width_height()

            # Set up pygame window
            if self.window is None:
                pygame.init()
                self.clock = pygame.time.Clock()

                # Set window size to figure's size in pixels
                window_size = tuple(map(int, fig.get_size_inches() * fig.dpi))
                self.window = pygame.display.set_mode(window_size)
                pygame.display.set_icon(Surface((0, 0)))
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
            raise ValueError(f"Invalid rednering mode: {mode}")
        


    def render_simulation(self, ax: plt.Axes) -> None:
        """Render the simulation view with entities."""

        colormap = cm.get_cmap("RdYlGn")
        # define normalization for unscaled utilities
        unorm = plt.Normalize(self.utility.lower, self.utility.upper)

        env = self.env
        
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

            # Plot BS ranges where UEs may connect or can receive at most 1MB/s
            #ax.scatter(*self.conn_isolines[bs], color="gray", s=3)
            #ax.scatter(*self.mb_isolines[bs], color="black", s=3)

        """
        for ue, utility in self.utilities.items():
            #utility = self.utility.unscale(utility)
            #color = colormap(unorm(utility))
        """

        for ue in env.users.values():
            ax.scatter(
                ue.x,
                ue.y,
                s=200,
                zorder=2,
                color="red",
                marker="o",
            )
            ax.annotate(ue.id, xy=(ue.x, ue.y), ha="center", va="center")

        """
        
        for bs in self.stations.values():
            for ue in self.connections[bs]:
                # color is connection's contribution to the UE's total utility
                share = self.datarates[(bs, ue)] / self.macro[ue]
                share = share * self.utility.unscale(self.utilities[ue])
                color = colormap(unorm(share))

                # add black background/borders for lines for visibility
                ax.plot(
                    [ue.point.x, bs.point.x],
                    [ue.point.y, bs.point.y],
                    color=color,
                    path_effects=[
                        pe.SimpleLineShadow(shadow_color="black"),
                        pe.Normal(),
                    ],
                    linewidth=3,
                    zorder=-1,
                )

        """
        

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


        # Remove simulation axis's ticks and spines
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xlim([0, self.env.width])
        ax.set_ylim([0, self.env.height])



    def render_dashboard(self, ax: plt.Axes) -> None:
        """"Render the info dashboard."""
        env = self.env





    def render_metrics(self, ax: plt.Axes) -> None:
        """Render metrics panel."""
        pass





    def close(self) -> None:
        """Close the renderer and release resources."""
        pygame.quit()
        self.window = None
        self.closed = True
