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

from metalore.visualization.symbols import BS_SYMBOL, SENSOR_SYMBOL, ISAC_SYMBOL


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
        fy = max(4 * env.height / dpi, 5.0)
        fig = plt.figure(figsize=(fx, fy))
        gs = fig.add_gridspec(
            ncols=2,
            nrows=5,
            width_ratios=(4, 2),
            height_ratios=(2, 3, 3, 3, 3),
            hspace=0.5,
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
        tx_queue_ax = fig.add_subplot(gs[3, 1])
        mec_queue_ax = fig.add_subplot(gs[4, 1])

        # Render each component
        self.render_simulation(env, sim_ax)
        self.render_metrics(env, metrics_ax)
        self.render_bw_allocation(env, bw_alloc_ax)
        self.render_comp_allocation(env, comp_alloc_ax)
        self.render_tx_queue_evolution(env,tx_queue_ax)
        self.render_mec_queue_evolution(env,mec_queue_ax)

        # Convert to image
        fig.align_ylabels((bw_alloc_ax, comp_alloc_ax))
        window_size = tuple(map(int, fig.get_size_inches() * fig.dpi))
        canvas = FigureCanvas(fig)
        canvas.draw()
        plt.close(fig)

        if mode == "rgb_array" or mode is None:
            data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            return data.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]

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
        active_isac_set = set(env.utilities_isac.keys())

        for ue in env.users.values():
            if ue not in active_ue_set:
                ax.scatter(
                    ue.x, ue.y,
                    s=200, zorder=1,
                    color="lightgray", marker="o", alpha=0.5,
                )
                ax.annotate(ue.id, xy=(ue.x, ue.y), ha="center", va="center",
                            color="gray", alpha=0.5)

        for isac in env.isacs.values():
            if isac not in active_isac_set:
                ax.scatter(
                    isac.x, isac.y,
                    s=200, zorder=1,
                    color="lightgray", marker=ISAC_SYMBOL, alpha=0.5,
                )
                ax.plot(
                    isac.x, isac.y,
                    marker=ISAC_SYMBOL,
                    markersize=10, markeredgewidth=0.1, color="gray", alpha=0.5, zorder=2
                )
                ax.annotate(isac.id, xy=(isac.x, isac.y), ha="center", va="center",
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

        for isac, utility in env.utilities_isac.items():
            utility = env.utility.unscale(utility)
            color = colormap(unorm(utility))
            mode_color = "blue" if isac.current_mode == 'UE' else "red"
            ax.plot(isac.x, isac.y, marker=ISAC_SYMBOL, markersize=10, markeredgewidth=0.1, color=mode_color, zorder=3)
            ax.annotate(isac.id, xy=(isac.x, isac.y), ha="center", va="center")

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
            
            for isac in env.connections_isac[bs]:
                color = "blue" if isac.current_mode == 'UE' else "red"
                # add black background/borders for lines for visibility
                ax.plot(
                    [isac.x, bs.x],
                    [isac.y, bs.y],
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
        aori_vals, aosi_vals, ue_data, sensor_data = [], [], 0.0, 0.0
        for job in env.job_tracker.completed_jobs:
            if job.job_type == 'UE':
                if job.aori is not None: aori_vals.append(job.aori)
                if job.aosi is not None: aosi_vals.append(job.aosi)
                ue_data += job.data_size
            else:
                sensor_data += job.data_size
        avg_aori = f"{sum(aori_vals)/len(aori_vals):.2f}" if aori_vals else "—"
        avg_aosi = f"{sum(aosi_vals)/len(aosi_vals):.2f}" if aosi_vals else "—"
        total_ue = f"{ue_data:.2f}"
        total_sensor = f"{sensor_data:.2f}"

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        table = ax.table(
            [[avg_aori, avg_aosi], [total_ue, total_sensor]],
            rowLabels=["Avg", "Throughput"],
            colLabels=["AoRI", "AoSI"],
            cellLoc="center",
            edges="B",
            loc="upper center",
            bbox=[0.0, -0.25, 1.0, 1.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

    def render_bw_allocation(self, env, ax: plt.Axes) -> None:
        bw_splits = env.metrics.step_totals['bw_split']
        time = np.arange(len(bw_splits))
        if env.config['environment']['num_ues'] != 0:
            ax.plot(time, bw_splits, linewidth=1, color="blue", label="UE")
        else:
            ax.plot(time, bw_splits, linewidth=1, color="purple", label="ISAC")
        ax.plot(time, 1 - np.array(bw_splits), linewidth=1, color="green", label="Sensor")

        ax.set_xlabel("Time")
        ax.set_ylabel("BW Allocation")
        ax.set_xlim([0.0, env.EP_MAX_TIME])
        ax.set_ylim([0.0, 1.0])
        ax.legend(loc="upper right", fontsize=8)

    def render_comp_allocation(self, env, ax: plt.Axes) -> None:
        comp_splits = env.metrics.step_totals['comp_split']
        time = np.arange(len(comp_splits))
        if env.config['environment']['num_ues'] != 0:
            ax.plot(time, comp_splits, linewidth=1, color="blue", label="UE")
        else:
            ax.plot(time, comp_splits, linewidth=1, color="purple", label="ISAC")
        ax.plot(time, 1 - np.array(comp_splits), linewidth=1, color="green", label="Sensor")

        ax.set_xlabel("Time")
        ax.set_ylabel("Comp. Allocation")
        ax.set_xlim([0.0, env.EP_MAX_TIME])
        ax.set_ylim([0.0, 1.0])
        ax.legend(loc="upper right", fontsize=8)
    
    def render_tx_queue_evolution(self, env, ax: plt.Axes) -> None:
        ue_queue = env.metrics.step_totals['ue_tx_queue_jobs']
        sensor_queue = env.metrics.step_totals['sensor_tx_queue_jobs']
        isac_queue = env.metrics.step_totals['isac_tx_queue_jobs']

        time = np.arange(len(ue_queue))

        if env.config['environment']['num_ues'] != 0:
            ax.plot(time, ue_queue, linewidth=1, color="blue", label="UE")
        else:
            ax.plot(time, isac_queue, linewidth=1, color="purple", label="ISAC")

        ax.plot(time, sensor_queue, linewidth=1, color="green", label="Sensor")
            
        ax.set_xlabel("Time")
        ax.set_ylabel("TxQueue Jobs")
        ax.set_xlim([0.0, env.EP_MAX_TIME])
        
        max_val = max(max(ue_queue) if ue_queue else 0, max(sensor_queue) if sensor_queue else 0)
        ax.set_ylim([0.0, max(10, max_val * 1.1)]) 
        
        ax.legend(loc="upper right", fontsize=8)

    def render_mec_queue_evolution(self, env, ax: plt.Axes) -> None:
        communication_queue = env.metrics.step_per_bs['ue_proc_queue_jobs']
        sensing_queue = env.metrics.step_per_bs['sensor_proc_queue_jobs']

        if not communication_queue:
            return
        
        num_steps = len(next(iter(communication_queue.values())))
        if num_steps == 0:
            return

        time = np.arange(num_steps)
        total_ue_mec = np.zeros(num_steps)
        total_sensor_mec = np.zeros(num_steps)
        
        for q_list in communication_queue.values():
            total_ue_mec += np.array(q_list)
        
        for q_list in sensing_queue.values():
            total_sensor_mec += np.array(q_list)

        
        ax.plot(time, total_ue_mec, linewidth=1, color="blue" if env.config['environment']['num_ues'] != 0 else 'purple', label="Communication")
        ax.plot(time, total_sensor_mec, linewidth=1, color="green", label="Sensing")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("MEC Queue Jobs")
        ax.set_xlim([0.0, env.EP_MAX_TIME])
        
        max_val = max(np.max(total_ue_mec), np.max(total_sensor_mec))
        ax.set_ylim([0.0, max(10, max_val * 1.1)]) 
        
        ax.legend(loc="upper right", fontsize=8)

    def close(self) -> None:
        """Close the renderer and release resources."""
        pygame.quit()
        self.window = None
        self.closed = True
