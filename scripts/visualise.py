# TODO!

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import wilcoxon

# helpers

## statistical tests 

def single_sig_label(
    scores_a, scores_b,
    sig_threshold = .001,
    labels_dict = {"greater":"˅","less":"˄"}
):
    lab=None
    if int(
            wilcoxon(
                np.array(scores_a), 
                np.array(scores_b), 
                alternative="greater"
            ).pvalue < sig_threshold
        ):
        lab = labels_dict["greater"]
    elif int(
            wilcoxon(
                np.array(scores_a), 
                np.array(scores_b),
                alternative="less"
            ).pvalue < sig_threshold
        ):
        lab = labels_dict["less"]
    
    return lab

def get_sigs(
    df_scores,
    comp_columns:list,
    col_prefix:str,
    split_by:tuple=None,
    ref_column:str="human",
    sig_threshold = .001,
    labels_dict = {"greater":"˅","less":"˄"}
):
    ref_vals = df_scores[f"{col_prefix}{ref_column}"]

    if not split_by is None:
        return [
            get_sigs(
                df_scores[df_scores[split_by[0]]==s],
                comp_columns=comp_columns,
                col_prefix=col_prefix,
                ref_column=ref_column,
                sig_threshold=sig_threshold,
                labels_dict=labels_dict,
            )
            for s in split_by[1]
        ]
    else:
        other_vals = [
            df_scores[f"{col_prefix}{comp_col}"]
            for comp_col in comp_columns
        ]
        return [
            single_sig_label(
                ref_vals, m_vals,
                sig_threshold=sig_threshold,
                labels_dict=labels_dict,
            )
            for m_vals in other_vals
        ]

## confidence intervals

def bootstrap_CI(
    data, 
    nbr_draws=1000,
    aggregator = np.nanmean
):
    if aggregator is None:
        aggregator = np.nanmean
        
    means = np.zeros(nbr_draws)
    data = np.array(data)

    for n in range(nbr_draws):
        indices = np.random.randint(0, len(data), len(data))
        data_tmp = data[indices] 
        means[n] = aggregator(data_tmp)

    return [np.nanpercentile(means, 2.5),np.nanpercentile(means, 97.5)]

def median_bootstrap_CI(
    data,
    **kwargs,
):
    return bootstrap_CI(
        data=data,
        aggregator=np.nanmedian,
        **kwargs
    )

# plotting functions

def single_barplot(
    ax,
    x_plot,
    heights,
    errors,
    colors=None,
    patterns=None,
    labels=None,
    annotations=None,
    edgecolor="k",
    bar_width=.8,
    capsize=8,
    fontsize=22,
):
    if colors is None:
        colors = [None]*len(x_plot)
    if patterns is None:
        patterns = [None]*len(x_plot)
    if annotations is None:
        annotations = [None]*len(x_plot)
    if labels is None:
        labels = [None]*len(x_plot)

    for k, h in enumerate(heights):
        if not errors[k] is None:
            if hasattr(errors[k], '__len__'):
                yerr = [
                    [np.abs(ci-heights[k])]
                    for ci in errors[k]
                ]
            else:
                yerr=errors[k]
        else:
            yerr=None

        ax.bar(
            x=x_plot[k],
            height=h,
            yerr=yerr,
            width=bar_width,
            hatch=patterns[k],
            color=colors[k],
            label=labels[k],
            capsize=capsize,
            edgecolor=edgecolor,
        )

        if not annotations[k] is None and annotations[k]:
            if not yerr is None:
                if len(yerr)==2:
                    y_value = h+yerr[1]
                else:
                    y_value = h+yerr
            else:
                y_value = h
            
            x_value = x_plot[k] 
            space = 1
            label = annotations[k] if type(annotations[k])==str else f"*"
            ax.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom', fontsize=fontsize)
    
    return ax
