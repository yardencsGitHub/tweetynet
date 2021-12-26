import seaborn as sns


def color_palette_from_labelmap(labelmap, palette=None, desat=None):
    n_colors = len(labelmap)
    pal_tmp = sns.color_palette(
        palette=palette,
        n_colors=n_colors,
        desat=desat,
        as_cmap=False
    )
    return {
        label: color
        for label, color in zip(
            sorted(labelmap.keys()), pal_tmp
        )
    }
