import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

class Figure:
    def __init__(self, fig_size=540, ratio=2, dpi=300, subplots=(1, 1),
                 width_ratios=None, height_ratios=None, 
                 hspace=None, wspace=None,
                 ts=2, pad=0.2, sw=0.2,
                 minor_ticks=True,
                 theme='dark', color=None, ax_color=None,
                 grid=True):

        fig_width, fig_height = fig_size * ratio / dpi, fig_size / dpi
        fs = np.sqrt(fig_width * fig_height)

        self.fs = fs
        self.fig_size = fig_size
        self.ratio = ratio
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.subplots = subplots
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.hspace = hspace
        self.wspace = wspace

        self.grid = grid
        self.ts = ts
        self.pad = pad
        self.sw = sw
        self.minor_ticks = minor_ticks

        self.fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        self.dpi = dpi

        # theme can only be dark or default, make a raiserror
        if not theme in ['dark', 'default']:
            raise ValueError('Theme must be "dark" or "default".')

        self.theme = theme  # This is set but not used in your provided code
        if theme == "dark":
            self.color = '#222222'
            self.ax_color = 'w'
            self.fig.patch.set_facecolor(self.color)
            plt.rcParams.update({"text.color": self.ax_color})
        else:
            self.color = 'w'
            self.ax_color = 'k'
            self.fig.patch.set_facecolor(self.color)
            plt.rcParams.update({"text.color": self.ax_color})

        if color is not None:
            self.color = color
        if ax_color is not None:
            self.ax_color = ax_color

        # GridSpec setup
        gs = mpl.gridspec.GridSpec(
            nrows=subplots[0], ncols=subplots[1], figure=self.fig,
            width_ratios=width_ratios or [1] * subplots[1],
            height_ratios=height_ratios or [1] * subplots[0],
            hspace=hspace, wspace=wspace
        )

        # Creating subplots
        self.axes = []
        for i in range(subplots[0]):
            row_axes = []
            for j in range(subplots[1]):
                ax = self.fig.add_subplot(gs[i, j])
                row_axes.append(ax)
            self.axes.append(row_axes)

        for i in range(subplots[0]):
            for j in range(subplots[1]):
                ax = self.axes[i][j]

                ax.set_facecolor(self.color)

                for spine in ax.spines.values():
                    spine.set_linewidth(fs * sw)
                    spine.set_color(self.ax_color)

                if grid:

                    ax.grid(
                        which="major",
                        linewidth=fs * sw*0.5,
                        color=self.ax_color,
                        alpha=0.25
                    )

                ax.tick_params(
                    axis="both",
                    which="major",
                    labelsize=ts * fs,
                    size=fs * sw*5,
                    width=fs * sw*0.9,
                    pad= pad * fs,
                    top=True,
                    right=True,
                    labelbottom=True,
                    labeltop=False,
                    direction='inout',
                    color=self.ax_color,
                    labelcolor=self.ax_color
                )

                if minor_ticks == True:
                    ax.minorticks_on()

                    ax.tick_params(axis='both', which="minor", 
                    direction='inout',
                    top=True,
                    right=True,
                    size=fs * sw*2.5, 
                    width=fs * sw*0.8,
                    color=self.ax_color)

        self.axes_flat = [ax for row in self.axes for ax in row]

        if hspace == 0:
            for i in range(subplots[0] - 1):
                self.axes[i][0].tick_params(labelbottom=False)


    def customize_axes(self, ax, ylabel_pos='left', 
                       xlabel_pos='bottom',):

        if ylabel_pos == 'left':
            labelright_bool = False
            labelleft_bool = True
        elif ylabel_pos == 'right':
            labelright_bool = True
            labelleft_bool = False

        if xlabel_pos == 'bottom':
            labeltop_bool = False
            labelbottom_bool = True
        elif xlabel_pos == 'top':
            labeltop_bool = True
            labelbottom_bool = False
        

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.ts * self.fs,
            size=self.fs * self.sw*5,
            width=self.fs * self.sw*0.9,
            pad= self.pad * self.fs,
            top=True,
            right=True,
            labelbottom=labelbottom_bool,
            labeltop=labeltop_bool,
            labelright=labelright_bool,
            labelleft=labelleft_bool,
            direction='inout',
            color=self.ax_color,
            labelcolor=self.ax_color
        )

        if self.minor_ticks == True:
            ax.minorticks_on()

            ax.tick_params(axis='both', which="minor", 
            direction='inout',
            top=True,
            right=True,
            size=self.fs * self.sw*2.5, 
            width=self.fs * self.sw*0.8,
            color=self.ax_color)

        ax.set_facecolor(self.color)

        for spine in ax.spines.values():
            spine.set_linewidth(self.fs * self.sw)
            spine.set_color(self.ax_color)

        if self.grid:

            ax.grid(
                which="major",
                linewidth=self.fs * self.sw*0.5,
                color=self.ax_color,
                alpha=0.25
            )

        return ax

    def save(self, path, bbox_inches=None, pad_inches=None):

        self.fig.savefig(path, dpi=self.dpi, bbox_inches=bbox_inches, pad_inches=None)

        self.path = path

    def check_saved_image(self):

        if not hasattr(self, 'path'):
            raise ValueError('Figure has not been saved yet.')


        with Image.open(self.path) as img:
            print(img.size)
            return
        
    def show_image(self):

        if not hasattr(self, 'path'):
            raise ValueError('Figure has not been saved yet.')
        
        with Image.open(self.path) as img:
            img.show()
            return













def initialize_figure(
    fig_size=540,
    ratio=1.5,
    fig_w=None, fig_h=None,
    subplots=(1, 1), grid=True, 
    lw=0.015, ts=2, theme=None,
    pad=0.5,
    color='#222222',
    dpi=300,
    sw=0.15,
    wr=None, hr=None, hmerge=None, wmerge=None,
    ylabel='bottom',
    layout='constrained',
    hspace=None, wspace=None,
    tick_direction='inout',
    minor=True,
    top_bool=True,
    projection=None
):
    """
    Initialize a Matplotlib figure with a specified size, aspect ratio, text size, and theme.

    Parameters:
    fig_size (float): The size of the figure.
    ratio (float): The aspect ratio of the figure.
    text_size (float): The base text size for the figure.
    subplots (tuple): The number of subplots, specified as a tuple (rows, cols).
    grid (bool): Whether to display a grid on the figure.
    theme (str): The theme for the figure ("dark" or any other string for a light theme).

    Returns:
    fig (matplotlib.figure.Figure): The initialized Matplotlib figure.
    ax (list): A 2D list of axes for the subplots.
    fs (float): The scaling factor for the figure size.
    """
    if fig_w is None:
        fig_w = fig_size * ratio
        fig_h = fig_size

    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    fig_size = fig_width * fig_height
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,  # Default dpi, will adjust later for saving
        layout=layout,
    )

    if wr is None:
        wr_ = [1] * subplots[1]
    else:
        wr_ = wr
    if hr is None:
        hr_ = [1] * subplots[0]
    else:
        hr_ = hr
    

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig, width_ratios=wr_, height_ratios=hr_, hspace=hspace, wspace=wspace)


    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    if theme == "dark":
        fig.patch.set_facecolor(color)
        plt.rcParams.update({"text.color": "white"})

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            
            if hmerge is not None:
                if i in hmerge:
                    ax[i][j] = fig.add_subplot(gs[i, :])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            elif wmerge is not None:
                if j in wmerge:
                    ax[i][j] = fig.add_subplot(gs[:, j])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            else:
                ax[i][j] = fig.add_subplot(gs[i, j], projection=projection)

            if theme == "dark":
                ax[i][j].set_facecolor(color)
                ax[i][j].tick_params(colors="white")
                ax[i][j].spines["bottom"].set_color("white")
                ax[i][j].spines["top"].set_color("white")
                ax[i][j].spines["left"].set_color("white")
                ax[i][j].spines["right"].set_color("white")
                ax[i][j].xaxis.label.set_color("white")
                ax[i][j].yaxis.label.set_color("white")

            #ax[i][j].xaxis.set_tick_params(which="minor", bottom=False)

            if grid:
                ax[i][j].grid(
                    which="major",
                    linewidth=fs * lw,
                    color="white" if theme == "dark" else "black",
                )
            for spine in ax[i][j].spines.values():
                spine.set_linewidth(fs * sw)

            if ylabel == 'bottom':
                labeltop_bool = False
                labelbottom_bool = True
            elif ylabel == 'top':
                labeltop_bool = True
                labelbottom_bool = False
                ax[i][j].xaxis.set_label_position('top')

            else:
                labeltop_bool = True
                labelbottom_bool = True
                ax[i][j].xaxis.set_label_position('both')

            
            ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=ts * fs,
                size=fs * sw*2,
                width=fs * sw,
                pad= pad * fs,
                top=top_bool,
                labelbottom=labelbottom_bool,
                labeltop=labeltop_bool,
                right=top_bool,
                direction=tick_direction
            )

            if minor:
                ax[i][j].minorticks_on()
                ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=ts * fs,
                size=fs * sw*4,
                width=fs * sw,
                pad= pad * fs,
                top=top_bool,
                labelbottom=labelbottom_bool,
                labeltop=labeltop_bool,
                right=top_bool,
                direction=tick_direction
                )
                ax[i][j].tick_params(axis='both', which="minor", 
                direction=tick_direction,
                top=top_bool,
                right=top_bool,
                size=fs * sw*2.5, width=fs * sw,)

    if hmerge is not None:
        for k in hmerge:
            for l in range(1, subplots[1]):
                fig.delaxes(ax[k][l])

    if wmerge is not None:
        for k in wmerge:
            for l in range(1, subplots[0]):
                fig.delaxes(ax[l][k])
            
    
    return fig, ax, fs, gs
