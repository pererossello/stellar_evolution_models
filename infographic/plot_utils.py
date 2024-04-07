import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
from matplotlib.path import Path
import PIL

from astropy import units as u
from astropy.constants import k_B, h, c

import utils as ut

mpl.rcParams["font.family"] = "monospace"


def initialize_figure_(
    fig_size=540,
    ratio=1.5,
    fig_w=None,
    fig_h=None,
    subplots=(1, 1),
    grid=True,
    lw=0.015,
    ts=2,
    theme=None,
    pad=0.5,
    color="#222222",
    dpi=300,
    sw=0.15,
    wr=None,
    hr=None,
    hmerge=None,
    wmerge=None,
    ylabel="bottom",
    layout="none",
    hspace=None,
    wspace=None,
    tick_direction="inout",
    minor=True,
    top_bool=True,
    projection=None,
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
    fig.patch.set_facecolor(color)
    plt.rcParams.update({"text.color": "white"})

    subplots = (4, 3)
    wr_ = [1] * subplots[1] if wr is None else wr
    hr_ = [1] * subplots[0] if hr is None else hr
    gs = mpl.gridspec.GridSpec(
        subplots[0],
        subplots[1],
        figure=fig,
        width_ratios=wr_,
        height_ratios=hr_,
        hspace=hspace,
        wspace=wspace,
    )

    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    ax[0][0] = fig.add_subplot(gs[0:2, 0])
    ax[1][0] = fig.add_subplot(gs[2:4, 0])
    for i in range(subplots[0]):
        ax[i][1] = fig.add_subplot(gs[i, 1])
    ax[0][2] = fig.add_subplot(gs[0:2, 2])
    ax[1][2] = fig.add_subplot(gs[2:4, 2])
    # remove ax[0][2]
    fig.delaxes(ax[0][2])

    # add new ax to figure
    ax_time = fig.add_axes([0.4025, 0.9015, 0.22, 0.05])

    axs = [
        ax[0][0],
        ax[1][0],
        ax[0][1],
        ax[1][1],
        ax[2][1],
        ax[3][1],
        ax[0][2],
        ax[1][2],
        ax_time,
    ]

    for ax_ in [ax[0][1], ax[1][1], ax[2][1]]:
        ax_.set_xticklabels([])

    for ax_ in axs:
        ax_.set_facecolor(color)
        ax_.tick_params(colors="white")
        for spine in ax_.spines.values():
            spine.set_color("white")
        ax_.xaxis.label.set_color("white")
        ax_.yaxis.label.set_color("white")

        ax_.grid(
            which="major",
            linewidth=fs * lw,
            color="white",
        )

        for spine in ax_.spines.values():
            spine.set_linewidth(fs * sw)

        ax_.tick_params(
            axis="both",
            which="major",
            labelsize=ts * fs,
            size=fs * sw * 4,
            width=fs * sw,
            pad=pad * fs,
            top=top_bool,
            labelbottom=True,
            labeltop=False,
            right=top_bool,
            direction=tick_direction,
        )

    ax[0][0].xaxis.set_label_position("top")
    ax[0][0].xaxis.tick_top()

    ax[1][2].yaxis.set_label_position("right")
    ax[1][2].yaxis.tick_right()

    ax[0][2].set_xticklabels([])
    ax[0][2].set_yticklabels([])
    ax[0][2].tick_params(axis="both", size=0)

    ax_time.xaxis.tick_top()
    ax_time.xaxis.set_label_position("top")

    ax_time.set_yticklabels([])
    ax_time.tick_params(axis="y", size=0)

    return fig, ax, ax_time, fs


def plot_pies(ax, fs, tab, step=0, cmap="gnuplot", color=None, first=True):

    cmap = mpl.colormaps[cmap]
    M = 6
    colors = [cmap(i / M) for i in range(M)]

    comps_dic = {"cen": {}, "surf": {}}
    current_sum = np.zeros_like(tab["time"])
    for reg in comps_dic:
        element_groups = ut.read_element_groups(reg)
        comps_dic[reg]["pie"] = []
        for group in element_groups.values():
            group_sum = np.sum(
                [np.array(tab[element][step]) for element in group], axis=0
            )
            next_sum = current_sum + group_sum
            current_sum = next_sum
            comps_dic[reg]["pie"].append(np.sum(group_sum))

    pie1 = ax.pie(comps_dic["surf"]["pie"], radius=1, colors=colors, startangle=90)

    pie2 = ax.pie(
        comps_dic["cen"]["pie"],
        colors=colors,
        startangle=90,
        radius=0.6,
        wedgeprops=dict(linewidth=0, edgecolor="w"),
    )

    if first == True:
        # plot matplotlib circle
        centre_circle = plt.Circle(
            (0, 0), 0.6, fc="none", edgecolor="white", lw=0.2 * fs, zorder=3
        )

        ax.add_artist(centre_circle)

        centre_circle = plt.Circle(
            (0, 0), 1, fc="none", edgecolor="white", lw=0.2 * fs, zorder=3
        )

        ax.add_artist(centre_circle)

        ax.legend(
            element_groups.keys(),
            loc=(0.01, -0.145),
            fontsize=fs * 1.15,
            frameon=False,
            ncols=3,
        )

        ax.plot(
            [0.5 + 0.25, 1.05],
            [0.5, 0.7],
            c="w",
            lw=0.2 * fs,
            zorder=3,
            transform=ax.transAxes,
        )
        ax.text(
            0.92,
            0.675,
            "Core",
            fontsize=fs * 1.5,
            color="w",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )

        ax.plot(
            [0.825, 1.1],
            [0.75, 0.95],
            c="w",
            lw=0.2 * fs,
            zorder=3,
            transform=ax.transAxes,
        )
        ax.text(
            0.8,
            0.9,
            "Surface",
            fontsize=fs * 1.5,
            color="w",
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )

        ax.text(
            0.5,
            -0.22,
            "Composition",
            fontsize=fs * 1.5,
            color="w",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    return pie1, pie2


def plot_mid(
    ax_mid,
    fs,
    tab,
    steps,
    i=0,
    lw_mid=0.25,
    color="w",
    step=0,
    first=True,
    fact_lum=1000,
):

    time = tab["time"]

    for ax in ax_mid:
        dif = time[steps[i]] - time[0]
        ax.set_xlim(time[0], time[steps[i]] + dif * 0.035)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    color_mid = color
    color_mid = tab["color"][steps[0] : steps[i]]
    lw_mid = 0.4

    radius = tab["radius"]

    sc1 = ax_mid[0].scatter(
        time[steps[0] : steps[i]],
        radius[steps[0] : steps[i]],
        c=color_mid,
        s=lw_mid * fs,
        lw=0 * lw_mid * fs,
    )

    mass = tab["mass"]

    sc2 = ax_mid[1].scatter(
        time[steps[0] : steps[i]],
        mass[steps[0] : steps[i]],
        c=color_mid,
        s=lw_mid * fs,
        lw=0 * lw_mid * fs,
    )

    ax_mid[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    mass_0 = np.round(tab["mass"][0], 2)
    bool_m = np.abs(mass[steps[i]] - mass_0) < 1e-2

    lum = 10 ** np.array(tab["lg(L)"]) / fact_lum

    sc3 = ax_mid[2].scatter(
        time[steps[0] : steps[i]],
        lum[steps[0] : steps[i]],
        c=color_mid,
        s=lw_mid * fs,
        lw=0 * lw_mid * fs,
    )

    Teff = 10 ** np.array(tab["lg(Teff)"]) / 1000

    sc4 = ax_mid[3].scatter(
        time[steps[0] : steps[i]],
        Teff[steps[0] : steps[i]],
        c=color_mid,
        s=lw_mid * fs,
        lw=0 * lw_mid * fs,
    )

    ax_mid[3].set_xlabel("Age [Myr]", fontsize=fs * 1.5)

    labels = [r"R/R$_\odot$", r"M/M$_\odot$", r"L/L$_\odot$", r"$T_{\rm eff}$ [kK]"]
    for i, ax in enumerate(ax_mid):
        # ste labels on the right
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(labels[i], fontsize=fs * 1.5, rotation=270, labelpad=2 * fs)

    return [sc1, sc2, sc3, sc4]


def plot_rhoc_tc(ax, fs, tab, steps, i=0, ts_lab=1.5, color="r", first=True):

    lgTmin, lgTmax = 6.9, 9.5

    if first == True:

        lgTs = np.linspace(lgTmin, lgTmax, 100) * u.K

        R = 8314 * u.s ** (-2) * u.m**2 * u.K ** (-1)
        mu = 0.63

        # P_ideal = P_e,deg
        K0 = R / mu
        K1 = 1e7 * 2 ** (-5 / 3) * u.m**4 * u.kg ** (-2 / 3) * u.s ** (-2)
        ct1 = K0 / K1
        ct1_val = ct1.value
        ct1_val = 3 / 2 * np.log10(ct1_val)
        lgrhoc_ideal_edeg = (3 / 2) * lgTs.value + ct1_val

        # P_ideal = P_e,deg-r
        K2 = 1.24e10 * 2 ** (-4 / 3) * u.m**3 * u.kg ** (-1 / 3) * u.s ** (-2)
        ct2 = K0 / K2
        ct2_val = ct2.value
        ct2_val = 3 * np.log10(ct2_val)
        lgrhoc_ideal_edegr = (3) * lgTs.value + ct2_val

        # P_e,deg = P_e,deg-r
        ct3 = K2 / K1
        ct3_val = ct3.value
        ct3_val = 3 * np.log10(ct3_val)
        lgrhoc_edeg_edegr = np.full(lgTs.shape, ct3_val)

        # P_ideal = P_rad
        a = 8 * np.pi**5 * k_B**4 / (15 * h**3 * c**3)
        cte4 = a / (3 * K0)
        cte4_val = cte4.value
        cte4_val = np.log10(cte4_val)
        lgrhoc_rad = 3 * lgTs.value + cte4_val

        dif = np.abs(lgrhoc_ideal_edeg - lgrhoc_ideal_edegr)
        idx = np.argmin(dif)

        lgrhoc_ideal_edeg_edegr = np.concatenate(
            (lgrhoc_ideal_edeg[:idx], lgrhoc_ideal_edegr[idx:])
        )

        cmap = mpl.colormaps["binary"]

        vals = [0.2, 0.35, 0.6, 0.8]
        reverse_vals = [1 - val for val in vals]

        colors_fill = [cmap(i) for i in reverse_vals]

        ax.fill_between(
            lgTs.value,
            20,
            lgrhoc_ideal_edeg_edegr,
            color=colors_fill[3],
            alpha=1,
            zorder=-1,
            lw=0.1 * fs,
            edgecolor=colors_fill[3],
        )
        ax.fill_between(
            lgTs[:idx].value,
            lgrhoc_edeg_edegr[:idx],
            lgrhoc_ideal_edeg_edegr[:idx],
            color=colors_fill[2],
            alpha=1,
            zorder=-1,
            lw=0.1 * fs,
            edgecolor=colors_fill[2],
        )
        ax.fill_between(
            lgTs.value,
            lgrhoc_ideal_edeg_edegr,
            lgrhoc_rad,
            color=colors_fill[1],
            alpha=1,
            zorder=-1,
            lw=0.1 * fs,
            edgecolor=colors_fill[1],
        )
        ax.fill_between(
            lgTs.value,
            lgrhoc_rad,
            color=colors_fill[0],
            alpha=1,
            zorder=-1,
            lw=0.1 * fs,
            edgecolor=colors_fill[0],
        )

        ts = 1
        ax.text(
            6.95,
            10.5,
            "III. e$^{-}$ Relativistic Degeneracy",
            fontsize=ts * fs,
            color="k",
            ha="left",
            va="center",
        )
        ax.text(
            6.95,
            9.2,
            "II. e$^{-}$ Degeneracy",
            fontsize=ts * fs,
            color="k",
            ha="left",
            va="center",
        )
        ax.text(
            6.95, 6, "I. Ideal Gas", fontsize=ts * fs, color="w", ha="left", va="center"
        )
        ax.text(
            7.9,
            3.6,
            "IV. Radiation Pressure",
            fontsize=ts * fs,
            color="w",
            ha="left",
            va="center",
        )

    colors = tab["color"]

    Tc = tab["lg(Tc)"]
    rhoc = tab["lg(rhoc)"]

    line = ax.scatter(
        Tc[steps[0] : steps[i]],
        rhoc[steps[0] : steps[i]],
        lw=0 * fs,
        c=colors[steps[0] : steps[i]],
        s=0.4 * fs,
        zorder=1,
    )

    radius = np.log10(tab["radius"][:])
    rad_norm = (radius) / (np.max(radius))
    lg_rad = rad_norm[steps[i]]

    scat = ax.scatter(
        Tc[steps[i]],
        rhoc[steps[i]],
        s=15 * lg_rad * fs,
        c=color,
        marker="*",
        zorder=3,
        linewidths=0.075 * fs,
        edgecolor="k",
    )

    ax.set_ylim(2, 11)
    ax.set_xlim(lgTmin, lgTmax)

    ax.set_xlabel(r"log $T_c$ [K]", fontsize=ts_lab * fs)
    ax.set_ylabel(r"log $\rho_c$ [kg m$^{-3}$]", fontsize=ts_lab * fs)

    return line, scat


def plot_hr(ax, fs, tab, steps, i=0, ts_lab=1.5, first=True, color="w"):

    if first == True:

        ts = 1

        fcolor = "k"
        tcolor = "lightgrey"
        ms_verts = [
            (4.5, 5),  # Start point
            (4.2, 1),  # Control point for curve
            (3.6, -1),  # Control point for curve and end point of first Bézier curve
            (3.4, -3.7),  # Control point for curve
            (3.6, -1.5),
            (4.2, 0.3),  # End point
            (4.5, 5),
            (4.5, 5),
        ]
        le = len(ms_verts)

        ms_codes = [Path.MOVETO] + [Path.CURVE4] * (le - 2) + [Path.CLOSEPOLY]

        ms_path = Path(ms_verts, ms_codes)
        ms_patch = patches.PathPatch(
            ms_path, facecolor=fcolor, alpha=0.5, edgecolor="none", lw=0.5, zorder=0
        )
        ax.add_patch(ms_patch)
        ax.text(
            3.7,
            -2,
            "Main Sequence",
            fontsize=fs * ts,
            rotation=-40,
            ha="center",
            va="center",
            color=tcolor,
        )

        g_verts = [
            (4.1, 2),  # Start point
            (3.6, 0.8),  # Control point for curve
            (3.1, 3),
            (3.6, 2.5),  # Control point for curve and end point of first Bézier curve
            (4.1, 2.1),
            (4.11, 2.1),
            (4.1, 2),
        ]
        le = len(g_verts)

        g_codes = [Path.MOVETO] + [Path.CURVE4] * (le - 1)

        g_path = Path(g_verts, g_codes)
        g_patch = patches.PathPatch(
            g_path, facecolor=fcolor, alpha=0.5, edgecolor="none", lw=0.5, zorder=0
        )
        ax.add_patch(g_patch)

        ax.text(
            3.7,
            2,
            "Giants",
            fontsize=fs * ts,
            rotation=0,
            ha="center",
            va="center",
            color=tcolor,
        )

        sg_verts = [
            (4.3, 5),  # Start point
            (4, 5.7),
            (3.8, 5.9),
            (3.45, 5.5),
            (3.4, 4),
            (3.6, 3),
            (4, 3.7),
            (4.1, 3.7),
            (4.3, 4.2),
            (4.3, 5),
            (4.3, 5),
        ]
        le = len(sg_verts)

        sg_codes = [Path.MOVETO] + [Path.CURVE4] * (le - 1)

        sg_path = Path(sg_verts, sg_codes)
        sg_patch = patches.PathPatch(
            sg_path, facecolor=fcolor, alpha=0.5, edgecolor="none", lw=0.5, zorder=0
        )
        ax.add_patch(sg_patch)

        ax.text(
            3.9,
            4.4,
            "Super Giants",
            fontsize=fs * ts,
            rotation=0,
            ha="center",
            va="center",
            color=tcolor,
        )

        xo, yo = 4, -3
        first = (1, 5)
        wd_verts = [
            first,
            (0.8, 5),
            (0.5, 1),
            (0, 1),
            (0, 0),
            (0.5, 0),
            (0.8, 2),
            first,
            # first,
        ]

        scale = 0.5
        wd_verts = [scale * np.array((x, y)) for x, y in wd_verts]
        wd_verts = [(xo + x, yo + y) for x, y in wd_verts]
        print(wd_verts)

        le = len(wd_verts)

        wd_codes = [Path.MOVETO] + [Path.CURVE4] * (le - 2) + [Path.CLOSEPOLY]

        wd_path = Path(wd_verts, wd_codes)
        wd_patch = patches.PathPatch(
            wd_path, facecolor=fcolor, alpha=0.5, edgecolor="none", lw=0.5, zorder=0
        )
        ax.add_patch(wd_patch)
        ax.text(
            4.4,
            -2.4,
            "White Dwarfs",
            fontsize=fs * ts,
            rotation=-40,
            ha="center",
            va="center",
            color=tcolor,
        )

        ax.invert_xaxis()

    colors = tab["color"]
    Teff = tab["lg(Teff)"]
    ele = tab["lg(L)"]

    line = ax.scatter(
        Teff[steps[0] : steps[i]],
        ele[steps[0] : steps[i]],
        lw=0 * fs,
        s=0.5 * fs,
        c=colors[steps[0] : steps[i]],
        zorder=2,
    )

    radius = np.log10(tab["radius"][:])
    rad_norm = (radius) / (np.max(radius))
    lg_rad = rad_norm[steps[i]]
    scat = ax.scatter(
        Teff[steps[i]],
        ele[steps[i]],
        s=15 * lg_rad * fs,
        c=color,
        zorder=4,
        marker="*",
        linewidths=0.075 * fs,
        edgecolor="k",
    )

    ax.set_xlim(np.log10(40000), np.log10(2000))
    ax.set_ylim(np.log10(1e-4), np.log10(1e6))

    ax.set_xlabel(r"log T$_{\rm eff}$ [K]", fontsize=ts_lab * fs)
    ax.set_ylabel(r"log L/L$_\odot$", fontsize=ts_lab * fs)

    return line, scat


def png_to_mp4(
    fold,
    title="video",
    fps=36,
    digit_format="04d",
    res=None,
    resize_factor=1,
    custom_bitrate=None,
    extension=".jpg",
):

    # Get a list of all .png files in the directory
    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    if not files:
        raise ValueError("No PNG files found in the specified folder.")

    im = PIL.Image.open(os.path.join(fold, files[0]))
    resx, resy = im.size

    if res is not None:
        resx, resy = res
    else:
        resx = int(resize_factor * resx)
        resy = int(resize_factor * resy)
        resx += resx % 2  # Ensuring even dimensions
        resy += resy % 2

    basename = os.path.splitext(files[0])[0].split("_")[0]

    ffmpeg_path = "ffmpeg"
    abs_path = os.path.abspath(fold)
    parent_folder = os.path.dirname(abs_path) + os.sep
    output_file = os.path.join(parent_folder, f"{title}.mp4")

    crf = 10  # Lower for higher quality, higher for lower quality
    bitrate = custom_bitrate if custom_bitrate else "5000k"
    preset = "slow"
    tune = "film"

    command = f'{ffmpeg_path} -y -r {fps} -i {os.path.join(fold, f"{basename}_%{digit_format}{extension}")} -c:v libx264 -profile:v high -crf {crf} -preset {preset} -tune {tune} -b:v {bitrate} -pix_fmt yuv420p -vf scale={resx}:{resy} {output_file}'

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error during video conversion:", e)


# def png_to_mp4(
#     fold, title="video", fps=36, digit_format="04d", res=(1920, 1080), basename=None
# ):

#     # Get a list of all .mp4 files in the directory
#     files = [f for f in os.listdir(fold) if f.endswith(".png")]
#     files.sort()

#     if basename is None:
#         name = os.path.splitext(files[0])[0]
#         basename = name.split("_")[0]
#         basename = basename + "_"
#     else:
#         basename = basename

#     ffmpeg_path = "ffmpeg"
#     framerate = fps
#     # framerate = 1
#     abs_path = os.path.abspath(fold)
#     parent_folder = os.path.dirname(abs_path) + "\\"
#     output_file = parent_folder + "{}.mp4".format(title, framerate)
#     crf = 18
#     bitrate = "10000k"  # 5 Mbps, adjust as needed
#     preset = "veryslow"  # Adjust as per your patience; slower will be better quality but take longer
#     tune = "film"  # Especially if your content is more graphical/animated

#     command = f"{ffmpeg_path} -y -r {framerate} -i {fold}{basename}%{digit_format}.png -c:v libx264 -crf {crf} -preset {preset} -tune {tune} -pix_fmt yuv420p -vf scale={res[0]}:{res[1]} {output_file}"

#     subprocess.run(command, shell=True)
