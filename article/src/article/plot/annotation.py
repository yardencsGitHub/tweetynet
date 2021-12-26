"""functions for plotting annotations for vocalizations"""
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np


def plot_segments(onsets,
                  offsets,
                  labels,
                  palette,
                  y=0.5,
                  seg_height=0.1,
                  ax=None,
                  patch_kwargs=None):
    """plot segments as rectangles on an axis.
    Used to visualize audio that is segmented into events,
    e.g. annotated vocalizations.

    Creates a collection of ``matplotlib.patch.Rectangle``s
    with the specified `onsets` and `offsets`
    all at height `y` and places them on the axes `ax`.

    Parameters
    ----------
    onsets : numpy.ndarray
        onset times of segments
    offsets : numpy.ndarray
        offset times of segments
    labels : numpy.ndarray
        of string labels for each segment
    palette: dict
        that maps unique string labels to colors
    y : float, int
        height on y-axis at which segments should be plotted.
        Default is 0.5.
    seg_height : float
        height of rectangles drawn to represent segments.
        Default is 0.1.
    ax : matplotlib.axes.Axes
        axes on which to plot segment. Default is None,
        in which case a new Axes instance is created
    patch_kwargs : dict
        keyword arguments passed to the `PatchCollection`
        that represents the segments. Default is None.
    """
    if patch_kwargs is None:
        patch_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots
    segments = [
        Rectangle(xy=(on, y), height=seg_height, width=off-on)
        for on, off in zip(onsets, offsets)
    ]
    facecolors = [
        palette[label] for label in labels
    ]

    pc = PatchCollection(segments, facecolors=facecolors, **patch_kwargs, label='segment')
    ax.add_collection(pc)
    return pc


def plot_labels(labels,
                t,
                t_shift_label=0.01,
                y=0.6,
                ax=None,
                text_kwargs=None):
    """plot labels on an axis.

    Parameters
    ----------
    labels : list, numpy.ndarray
    t : numpy.ndarray
        times (in seconds) at which to plot labels
    t_shift_label : float
        amount (in seconds) that labels should be shifted to the left, for centering.
        Necessary because width of text box isn't known until rendering.
    y : float, int
        height on y-axis at which segments should be plotted.
        Default is 0.5.
    ax : matplotlib.axes.Axes
        axes on which to plot segment. Default is None,
        in which case a new Axes instance is created
    text_kwargs : dict
        keyword arguments passed to the `Axes.text` method
        that plots the labels. Default is None.

    Returns
    -------
    artists : list
        of matplotlib.Text instances for each label
    """
    if text_kwargs is None:
        text_kwargs = {}

    if ax is None:
        fig, ax = plt.subplots

    artists = []
    for label, t_lbl in zip(labels, t):
        t_lbl -= t_shift_label
        text = ax.text(t_lbl, y, label, **text_kwargs, label='label')
        artists.append(text)

    return artists


def annotation(annot,
               palette,
               tlim=None,
               y_segments=0.5,
               seg_height=0.1,
               y_labels=0.6,
               t_shift_label=0.01,
               patch_kwargs=None,
               text_kwargs=None,
               ax=None):
    """plot segments with labels, from annotation

    Parameters
    ----------
    annot : crowsetta.Annotation
        annotation that has segments to be plotted
        (the `annot.seq.segments` attribute)
    palette : dict
        that maps unique string labels to colors
    tlim : tuple, list
        limits of time axis (tmin, tmax) (i.e., x-axis).
        Default is None.
    y_segments : float
        height at which segments should be plotted.
        Default is 0.5 (assumes y-limits of 0 and 1).
    seg_height : float
        height of rectangles drawn to represent segments.
        Default is 0.1.
    y_labels : float
        height at which labels should be plotted.
        Default is 0.6 (assumes y-limits of 0 and 1).
    t_shift_label : float
        amount (in seconds) that labels should be shifted to the left, for centering.
        Necessary because width of text box isn't known until rendering.
    patch_kwargs : dict
        keyword arguments for `PatchCollection`.
        Passed to the function `plot_segments` that plots segments
        as a `PatchCollection` instance. Default is None.
    text_kwargs : dict
        keyword arguments for `matplotlib.axes.Axes.text`.
        Passed to the function `plot_labels` that plots labels
        using Axes.text method. Default is None.
    ax : matplotlib.axes.Axes
        axes on which to plot segments.
        Default is None, in which case
        a new figure with a single axes is created
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_ylim(0, 1)

    segment_centers = []
    for on, off in zip(annot.seq.onsets_s, annot.seq.offsets_s):
        segment_centers.append(
            np.mean([on, off])
        )
    plot_segments(onsets=annot.seq.onsets_s,
                  offsets=annot.seq.offsets_s,
                  labels=annot.seq.labels,
                  palette=palette,
                  y=y_segments,
                  seg_height=seg_height,
                  ax=ax,
                  patch_kwargs=patch_kwargs)

    if tlim:
        ax.set_xlim(tlim)
        tmin, tmax = tlim

        labels = []
        segment_centers_tmp = []
        for label, segment_center in zip(annot.seq.labels, segment_centers):
            if tmin < segment_center < tmax:
                labels.append(label)
                segment_centers_tmp.append(segment_center)
        segment_centers = segment_centers_tmp
    else:
        labels = annot.seq.labels

    segment_centers = np.array(segment_centers)
    plot_labels(labels=labels,
                t=segment_centers,
                t_shift_label=t_shift_label,
                y=y_labels,
                ax=ax,
                text_kwargs=text_kwargs)
