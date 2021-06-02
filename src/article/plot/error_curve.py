import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# max width in inches is 7.5
# https://journals.plos.org/ploscompbiol/s/figures
FIGSIZE = (7.5, 3.75)
DPI = 300


def across_animals(curve_df,
                   suptitle=None,
                   ax1_ylim=(0, 8),
                   ax2_ylim=(0.1, 0.65)):
    TRAIN_DUR_IND_MAP = {
        k: v for k, v in zip(
            sorted(curve_df['train_set_dur'].unique()),
            sorted(curve_df['train_set_dur_ind'].unique())
        )
    }

    fig = plt.figure(constrained_layout=True, figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(nrows=4, ncols=2, hspace=0.005)

    ax_arr = []
    ax_arr.append(fig.add_subplot(gs[0, 0]))
    ax_arr.append(fig.add_subplot(gs[1:, 0]))
    ax_arr.append(fig.add_subplot(gs[:, 1]))

    ax_arr = np.asarray(ax_arr)
    ax_arr[0].get_shared_x_axes().join(*ax_arr[:2].tolist())

    # for col in range(2):
    ax_arr[0].spines['bottom'].set_visible(False)
    ax_arr[1].spines['top'].set_visible(False)
    ax_arr[1].xaxis.tick_bottom()

    metric_list = ['avg_error', 'avg_segment_error_rate']
    ylabels = ['Frame error (%)', 'Segment error rate\n(edits per segment)']

    for col, (metric, ylabel) in enumerate(zip(metric_list, ylabels)):
        if col == 0:
            ax = ax_arr[1]
        else:
            ax = ax_arr[2]

        if col == 1:
            legend = 'full'
        else:
            legend = False

        sns.lineplot(x='train_set_dur_ind',
                     y=metric,
                     hue='animal_id',
                     data=curve_df,
                     ci='sd',
                     palette='colorblind',
                     linewidth=2,
                     ax=ax,
                     legend=legend)
        sns.lineplot(x='train_set_dur_ind',
                     y=metric,
                     linestyle='dashed',
                     color='k',
                     linewidth=4,
                     data=curve_df, ci=None, label='mean', ax=ax, legend=legend)

        ax.set_ylabel('')

        ax.set_xlabel('Training set duration (s)', fontsize=10)
        ax.set_xticks(list(TRAIN_DUR_IND_MAP.values()))
        ax.set_xticklabels(sorted(curve_df['train_set_dur'].unique().astype(int)), rotation=45)

    ax_arr[0].set_xticklabels([])
    ax_arr[0].set_xlabel('')

    # zoom-in / limit the view to different portions of the data
    ax_arr[0].set_ylim(40, 100)
    # ax_arr[1, 0].set_ylim(0, 14)
    ax_arr[1].set_ylim(ax1_ylim)
    ax_arr[2].set_ylim(ax2_ylim)

    bigax_col0 = fig.add_subplot(gs[:, 0], frameon=False)
    bigax_col1 = fig.add_subplot(gs[:, 1], frameon=False)
    labelpads = (2, 10)
    panel_labels = ['A', 'B']
    for ylabel, labelpad, panel_label, ax in zip(ylabels,
                                                 labelpads,
                                                 panel_labels,
                                                 [bigax_col0, bigax_col1]):
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.grid(False)
        ax.set_ylabel(ylabel, fontsize=10, labelpad=labelpad)
        ax.text(-0.2, 1., panel_label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='right')

    # get handles from right axes legend, then remove and re-create outside
    handles, _ = ax_arr[2].get_legend_handles_labels()
    # [ha.set_linewidth(2) for ha in handles ]
    ax_arr[2].get_legend().remove()
    bigax_col1.legend(handles=handles, bbox_to_anchor=(1.35, 1))

    fig.set_constrained_layout_pads(hspace=-0.05, wspace=0.0)

    if suptitle is not None:
        fig.suptitle(suptitle)

    return fig
