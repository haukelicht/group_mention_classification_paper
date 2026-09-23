import pandas as pd
import numpy as np
from pandas import DataFrame

from statsmodels.stats.proportion import proportion_confint
from scipy.cluster.hierarchy import linkage, leaves_list

from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
import matplotlib.patches as patches
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import colorsys

from IPython.display import Latex, display
import re
from typing import Optional, Literal, List, Tuple

from src.categories import *

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

plt.rcParams.update({
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Verdana",
        "Arial",
        "sans-serif",
    ],
    "font.monospace": [
        "Courier New",
        "Courier",
        "monospace"
    ],

    'figure.figsize': (7, 4),

    # Embed TrueType fonts in PDF/EPS
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Optional: fix sizes for reproducibility
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
})

# Normalize party family labels
family_map = {
    'prrp': 'Populist Radical-Right', 
    'green': 'Green',
    'con': 'Conservative',
    'sd': 'Social Democratic',
}

fam_hue_order = ['Populist Radical-Right', 'Green']

# use colors for diverging colormap (PRGn) to ensure consistency with heatmaps
norm = mcolors.Normalize(vmin=-1, vmax=1)
fam_col_palette = {
    'Populist Radical-Right': mcolors.to_hex(plt.cm.PRGn(norm(-0.7))), 
    'Green': mcolors.to_hex(plt.cm.PRGn(norm(0.7)))
}
all_fam_order = fam_hue_order.copy()
all_fam_order += ['Conservative', 'Social Democratic']
all_fam_palette = fam_col_palette.copy()
all_fam_palette.update({
    'Conservative': '#377eb8',      # Blue for conservatives
    'Social Democratic': '#e41a1c',       # Red for social democrats
})

all_fam_hue_order = all_fam_order.copy()

def normalize_family(x):
    if pd.isna(x):
        return None
    lx = str(x).strip().lower()
    return family_map.get(lx, None)


def increase_saturation(color, factor=1.5):
    """Increase saturation of a color while keeping hue and value the same."""
    # Convert to RGB then HSV
    rgb = mcolors.to_rgb(color)
    hsv = colorsys.rgb_to_hsv(*rgb)
    
    # Increase saturation, clamping to max 1.0
    new_saturation = min(hsv[1] * factor, 1.0)
    
    # Convert back to RGB
    new_rgb = colorsys.hsv_to_rgb(hsv[0], new_saturation, hsv[2])
    return mcolors.to_hex(new_rgb)

def darken_color(color, factor=0.8):
    """Darken a color by reducing its value component."""
    rgb = mcolors.to_rgb(color)
    hsv = colorsys.rgb_to_hsv(*rgb)
    
    # Decrease value (brightness)
    new_value = hsv[2] * factor
    
    new_rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], new_value)
    return mcolors.to_hex(new_rgb)

def intensify_color(color, sat_factor=1.3, val_factor=0.9):
    """Make color more intense by increasing saturation and slightly darkening."""
    rgb = mcolors.to_rgb(color)
    hsv = colorsys.rgb_to_hsv(*rgb)
    
    # Increase saturation and decrease value slightly
    new_saturation = min(hsv[1] * sat_factor, 1.0)
    new_value = hsv[2] * val_factor
    
    new_rgb = colorsys.hsv_to_rgb(hsv[0], new_saturation, new_value)
    return mcolors.to_hex(new_rgb)

# Complete color scheme setup
def blend_hex_colors(color1, color2, weight1=0.5):
    """Blend two hex colors and return hex result."""
    rgb1 = mcolors.to_rgb(color1)
    rgb2 = mcolors.to_rgb(color2)
    weight2 = 1 - weight1
    blended = tuple(weight1 * c1 + weight2 * c2 for c1, c2 in zip(rgb1, rgb2))
    return mcolors.to_hex(blended)


def show_color_palette(color_dict, title="Color Palette"):
    """Display colors as squares with labels."""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    square_size = 0.8
    spacing = 1.0
    
    for i, (label, color) in enumerate(color_dict.items()):
        # Create colored square
        square = patches.Rectangle((i * spacing, 0), square_size, square_size, 
                                 facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(square)
        
        # Add label below square
        ax.text(i * spacing + square_size/2, -0.2, label, 
               ha='center', va='top', fontsize=10, rotation=0)
        
        # Add hex code above square
        ax.text(i * spacing + square_size/2, square_size + 0.1, color, 
               ha='center', va='bottom', fontsize=8, family='monospace')
    
    ax.set_xlim(-0.2, len(color_dict) * spacing)
    ax.set_ylim(-0.5, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()


def show_colors_on_axis(ax, color_dict, title):
    square_size = 0.8
    spacing = 1.0
    
    for i, (label, color) in enumerate(color_dict.items()):
        square = patches.Rectangle((i * spacing, 0), square_size, square_size, 
                                 facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(square)
        ax.text(i * spacing + square_size/2, -0.2, label, 
               ha='center', va='top', fontsize=9)
        ax.text(i * spacing + square_size/2, square_size + 0.1, color, 
               ha='center', va='bottom', fontsize=7, family='monospace')
    
    ax.set_xlim(-0.2, len(color_dict) * spacing)
    ax.set_ylim(-0.5, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontweight='bold')


def latex_table(
        df: DataFrame, 
        position: str = "!th", 
        fontsize: Literal["normalsize", "small", "footnotesize", "scriptsize"] = "normalsize",
        multicolumn_cmidrules: Optional[List[Tuple[int, int, int]]] = None,
        resize: bool = False,
        landscape:  bool = False,
        **kwargs
    ) -> str:

    dflt_args = dict(
        position = position,
        na_rep = '',
        float_format = "%0.3f",
        sparsify = True,
        index_names = True,
        longtable = False,
        escape = False,
        multicolumn = True,
        multicolumn_format = "c",
        multirow = True,
        index=False
    )

    dflt_args.update(**kwargs)

    latex = df.to_latex(**dflt_args)

    # center the table float (if any) by default
    latex = re.sub(r'(\\begin\{table\}(?:\[[^\]]*\])?\n)', r'\1\\centering\n', latex, count=1)

    # add line space after clines
    latex = re.sub(r'^\\cline\{(\d+)-(\d+)\}.*(?=\n)', r'\\cmidrule(lr){\1-\2}', latex, flags=re.MULTILINE)
    
    # remove any midrule(s) before bottomrule as in this pattern
    latex = re.sub(r'^(\\cmidrule\(lr\)\{\d+-\d+\}\n)(\\bottomrule)', r'\2', latex, flags=re.MULTILINE)

    if multicolumn_cmidrules:
        latex_list = latex.splitlines()

        # get offset (number of line that starts with '\toprule')
        offset = next(i for i, line in enumerate(latex_list) if line.startswith(r'\toprule')) + 1
        for cols in multicolumn_cmidrules:
            row, start, end = cols
            latex_list.insert(row + offset, f'\\cmidrule(lr){{{start}-{end}}}')
        latex = '\n'.join(latex_list)

    if resize:
        resize_width = r"1.3\textheight" if landscape else r"\textwidth"
        tabular_start = latex.find(r'\begin{tabular}')
        tabular_end = latex.rfind(r'\end{tabular}')
        if tabular_start != -1 and tabular_end != -1:
            tabular_end += len(r'\end{tabular}')
            latex = (
                latex[:tabular_start]
                + fr"\resizebox{{{resize_width}}}{{!}}{{%" + "\n"
                + latex[tabular_start:tabular_end]
                + "\n}%"
                + latex[tabular_end:]
            )

    return latex


def format_table(df, align: str | list[str] = 'r', ls: bool = False, index=False, **kwargs): 
    n_index = len(df.index.names)
    index_col_spec = 'r' * n_index
    if isinstance(align, str):
        data_col_spec = align * len(df.columns)
    else:
        if len(align) != len(df.columns):
            raise ValueError("Length of align list must match number of columns in the DataFrame.")
        data_col_spec = align
    tab = latex_table(df, landscape=ls, column_format=index_col_spec+data_col_spec, index=index, **kwargs)
    return tab


def save_table(tab: str, file_name: str, dest, caption: Optional[str] = None):
    fp = dest / file_name
    with open(fp, 'w') as f:
        f.write(tab)
    if caption:
        fp_tex = dest / f"{fp.stem}-caption.tex"
        with open(fp_tex, 'w') as f:
            f.write(caption)


def save_figure(fig, file_name, caption, dest, bbox_inches='tight', pad_inches=0.1, transparent=False):
    fp = dest / file_name
    fig_to_save = fig.figure if hasattr(fig, 'figure') else fig
    fig_to_save.savefig(fp, bbox_inches=bbox_inches, pad_inches=pad_inches, transparent=transparent, dpi=300)
    fp_tex = dest / f"{fp.stem}-caption.tex"
    with open(fp_tex, 'w') as f:
        f.write(caption)


# Compute prevalence (share of mentions containing each attribute) and visualize

def binarize_column(col: pd.Series) -> pd.Series:
    """Convert a column to binary indicator (0/1)."""
    if pd.api.types.is_bool_dtype(col):
        return col
    if pd.api.types.is_numeric_dtype(col):
        if col.max() <= 1:
            return col >= 0.5
        return col > 0
    try:
        col_float = pd.to_numeric(col, errors='coerce')
        return col_float >= 0.5
    except Exception:
        return col.notna()

def compute_prevalence(df: pd.DataFrame, attrs: list, group_by=None) -> pd.DataFrame:
    """
    Compute prevalence of attributes with optional grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing attribute columns.
    attrs : list
        List of attribute column names to compute prevalence for.
    group_by : str | list | tuple | None
        Column name or list/tuple of column names to group by (e.g., 'party_family' or ['party_family', 'year']).
        If None, computes overall prevalence.

    Returns
    -------
    pd.DataFrame
        Columns: attribute, prevalence, ci_low, ci_high, n, plus any grouping columns if provided.
    """
    rows = []
    group_cols = []
    if group_by is None:
        group_cols = None
    elif isinstance(group_by, (list, tuple)):
        group_cols = list(group_by)
    else:
        group_cols = [group_by]

    if group_cols is None:
        # Overall prevalence
        for attr in attrs:
            if attr not in df.columns:
                continue
            bin_col = binarize_column(df[attr])
            n = len(bin_col)
            k = bin_col.sum()
            prevalence = k / n if n > 0 else 0

            if n > 0:
                low, high = proportion_confint(count=k, nobs=n, alpha=0.05, method='wilson')
            else:
                low, high = 0, 0

            rows.append({
                'attribute': attr,
                'prevalence': prevalence,
                'ci_low': low,
                'ci_high': high,
                'n': n
            })
    else:
        # Grouped prevalence (supports multiple grouping columns)
        for attr in attrs:
            if attr not in df.columns:
                continue
            for keys, sub in df.groupby(group_cols):
                bin_col = binarize_column(sub[attr])
                n = len(bin_col)
                k = bin_col.sum()
                if n == 0:
                    continue
                prevalence = k / n

                low, high = proportion_confint(count=k, nobs=n, alpha=0.05, method='wilson')

                record = {
                    'attribute': attr,
                    'prevalence': prevalence,
                    'ci_low': low,
                    'ci_high': high,
                    'n': n
                }
                if isinstance(keys, tuple):
                    for col, val in zip(group_cols, keys):
                        record[col] = val
                else:
                    record[group_cols[0]] = keys
                rows.append(record)

    if not rows:
        cols = ['attribute', 'prevalence', 'ci_low', 'ci_high', 'n']
        if group_cols:
            cols = ['attribute', *group_cols, 'prevalence', 'ci_low', 'ci_high', 'n']
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out.sort_values(by=['attribute'] + (group_cols if group_cols else []) + ['prevalence'], inplace=True)
    return out


def plot_prevalence_bars(
    data: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    attribute_order: list = None,
    hue_col: str = None,
    hue_order: list = None,
    palette: dict = None,
    xlim: tuple = (0, 0.32),
    bar_color: str = 'grey',
    fontsize: int = None
):
    """
    Plot horizontal bar chart of prevalence with confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Data with columns: attribute, prevalence, ci_low, ci_high, [hue_col if specified]
    ax : plt.Axes
        Matplotlib axes to plot on
    title : str
        Plot title
    attribute_order : list, optional
        Order of attributes on y-axis. If None, uses data order.
    hue_col : str, optional
        Column name for grouping/coloring bars (e.g., 'party_family')
    hue_order : list, optional
        Order of hue categories
    palette : dict, optional
        Color mapping for hue categories
    xlim : tuple
        X-axis limits (default: (0, 0.25))
    bar_color : str
        Bar color when hue_col is None (default: 'grey')
    """
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.axis('off')
        return

    data_plot = data.copy()

    # Set attribute ordering
    if attribute_order is None:
        if hue_col:
            # Order by mean prevalence across groups
            attribute_order = data_plot.groupby('attribute')['prevalence'].mean().sort_values(ascending=False).index.tolist()
        else:
            attribute_order = data_plot['attribute'].tolist()

    data_plot['attribute'] = pd.Categorical(
        data_plot['attribute'],
        categories=attribute_order,
        ordered=True
    )

    # set color of grid lines to light grey with alpha = 0.5
    ax.grid(color='lightgrey', alpha=0.5)

    # Plot bars
    if hue_col:
        # Grouped bars
        data_plot[hue_col] = pd.Categorical(data_plot[hue_col], categories=hue_order, ordered=True)
        data_plot = data_plot.sort_values(['attribute', hue_col])

        sns.barplot(
            data=data_plot,
            x='prevalence',
            y='attribute',
            hue=hue_col,
            hue_order=hue_order,
            order=attribute_order,
            orient='h',
            palette=palette,
            dodge=True,
            linewidth=1,
            edgecolor='w',
            errorbar=None,
            legend=False,
            ax=ax
        )

        # Add CI whiskers (need to resort for patch alignment)
        data_plot = data_plot.sort_values([hue_col, 'attribute'])
    else:
        # Single color bars
        sns.barplot(
            data=data_plot,
            x='prevalence',
            y='attribute',
            orient='h',
            color=bar_color,
            ax=ax
        )

    # Configure axes
    ax.set_xlim(xlim)
    ax.set_xlabel('Prevalence (share of mentions)')
    ax.set_ylabel('')
    ax.set_title(title, fontweight='bold')

    # Add 95% CI whiskers and text annotations
    for patch, (_, row) in zip(ax.patches, data_plot.iterrows()):
        width = patch.get_width()
        y_center = patch.get_y() + patch.get_height() / 2
        low = row['ci_low']
        high = row['ci_high']

        # Draw CI whisker
        ax.plot([low, high], [y_center, y_center], color='k', linewidth=1.2)

        # Add prevalence text annotation to the right of the upper CI bound
        fontsize = fontsize if fontsize else 7 if hue_col else 9
        ax.text(high + 0.002, y_center, f"{width:.3f}", va='center', fontsize=fontsize)


def compute_cooccurrence_stats(df: pd.DataFrame, attrs: list) -> pd.DataFrame:
    """
    For each attribute, compute how often it appears alone vs. with other attributes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing attribute columns
    attrs : list
        List of attribute column names
    
    Returns
    -------
    pd.DataFrame
        Columns: attribute, count_alone, count_with_others, total_with_attr, pct_alone
    """
    rows = []
    
    for attr in attrs:
        if attr not in df.columns:
            continue
        
        # Binarize the attribute column
        bin_col = binarize_column(df[attr])
        
        # Mentions where this attribute is present
        mentions_with_attr = bin_col[bin_col].index
        
        if len(mentions_with_attr) == 0:
            continue
        
        # For each mention with this attribute, count how many attributes it has in total
        df_attr_present = df.loc[mentions_with_attr].copy()
        
        # Binarize all attribute columns for these mentions
        attr_counts = pd.DataFrame()
        for a in attrs:
            if a in df.columns:
                attr_counts[a] = binarize_column(df_attr_present[a])
        
        # Count total attributes per mention
        total_attrs_per_mention = attr_counts.sum(axis=1)
        
        # Mentions where THIS attribute is the only one present
        alone = (total_attrs_per_mention == 1).sum()
        
        # Mentions where THIS attribute appears with others
        with_others = (total_attrs_per_mention > 1).sum()
        
        total = len(mentions_with_attr)
        pct_alone = (alone / total * 100) if total > 0 else 0
        
        rows.append({
            'attribute': attr,
            'count_alone': alone,
            'count_with_others': with_others,
            'total_with_attr': total,
            'pct_alone': pct_alone
        })
    
    result = pd.DataFrame(rows)
    return result.sort_values('pct_alone', ascending=False)

def compute_cooccurrence_breakdown(df: pd.DataFrame, attrs: list) -> pd.DataFrame:
    """
    For each attribute, compute prevalence when it appears alone vs. co-occurs with specific other attributes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing attribute columns
    attrs : list
        List of attribute column names
    
    Returns
    -------
    pd.DataFrame
        Columns: focal_attr, cooccur_with, count, total_with_focal, prevalence
        - When cooccur_with is 'alone': mentions where focal_attr is the only one
        - Otherwise: mentions where focal_attr co-occurs with that specific attribute
    """
    rows = []
    
    for focal_attr in attrs:
        if focal_attr not in df.columns:
            continue
        
        # Binarize the focal attribute
        bin_focal = binarize_column(df[focal_attr])
        mentions_with_focal = bin_focal[bin_focal].index
        
        if len(mentions_with_focal) == 0:
            continue
        
        total_with_focal = len(mentions_with_focal)
        df_focal = df.loc[mentions_with_focal].copy()
        
        # Binarize all attributes for these mentions
        attr_binarized = {}
        for a in attrs:
            if a in df.columns:
                attr_binarized[a] = binarize_column(df_focal[a])
        
        # Count total attributes per mention (including focal)
        total_attrs_per_mention = pd.DataFrame(attr_binarized).sum(axis=1)
        
        # Mentions where focal attribute is alone
        alone_mask = total_attrs_per_mention == 1
        count_alone = alone_mask.sum()
        
        rows.append({
            'focal_attr': focal_attr,
            'cooccur_with': 'alone',
            'count': count_alone,
            'total_with_focal': total_with_focal,
            'prevalence': count_alone / total_with_focal if total_with_focal > 0 else 0
        })
        
        # For mentions where focal appears with others, count co-occurrence with each other attribute
        cooccur_mask = total_attrs_per_mention > 1
        
        for other_attr in attrs:
            if other_attr == focal_attr or other_attr not in df.columns:
                continue
            
            # Count mentions where focal co-occurs with this specific other attribute
            other_bin = attr_binarized[other_attr]
            cooccur_count = (cooccur_mask & other_bin).sum()
            
            if cooccur_count > 0:
                rows.append({
                    'focal_attr': focal_attr,
                    'cooccur_with': other_attr,
                    'count': cooccur_count,
                    'total_with_focal': total_with_focal,
                    'prevalence': cooccur_count / total_with_focal if total_with_focal > 0 else 0
                })
    
    return pd.DataFrame(rows)

def plot_heatmap(
    x: pd.DataFrame,
    panel_groups: Tuple[List] = None,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    mask_diagonal: bool = True,
    cmin: float = None,
    cmap: Literal['PRGn', 'PiYG', 'YlOrRd'] = 'PRGn',
    clims: tuple = (-1, 1),
    clegend_title: str = None,
    annot: bool = True,
    fmt: str = '0.2f',
    annot_fontsize: float = 7,
    figsize_multiplier: tuple = (0.375, 0.45),
    linewidths: float = 1.0,
    linecolor: str = 'white',
    na_color: str = '#fbfbfb',
    xlabel_rotation: float = 30,
    xlabel_ha: str = 'left'
) -> tuple:
    """
    Create a heatmap for attribute association matrices.
    
    Parameters
    ----------
    x : pd.DataFrame
        Square data frame with values to plot
    cluster_rows : bool, default=False
        Whether to apply hierarchical clustering to rows
    cluster_cols : bool, default=False
        Whether to apply hierarchical clustering to columns
    panel_groups : tuple[list, list], optional
        Tuple of (row_groups, col_groups) where each is a list of index/column names
        to separate into panels. If provided, creates separate subplots for first
        and remaining groups. If None, creates single heatmap.
    cmap : str, default='PiYG'
        Colormap to use
    clims : tuple, default=(0, 1)
        Color limits (vmin, vmax)
    clegend_title : str, optional
        Title for the colorbar legend
    annot : bool, default=True
        Whether to annotate cells with values
    fmt : str, default='.2f'
        Format string for annotations
    figsize_multiplier : tuple, default=(0.7, 0.5)
        Multipliers for (width, height) based on data dimensions
    linewidths : float, default=0.5
        Width of cell borders
    linecolor : str, default='white'
        Color of cell borders
    xlabel_rotation : float, default=30
        Rotation angle for x-axis labels
    xlabel_ha : str, default='left'
        Horizontal alignment for x-axis labels
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : matplotlib.axes.Axes or list
        The axes object(s)
    """
    # Make a copy to avoid modifying original
    data = x.copy()
    
    # Step 2: Apply clustering if requested
    if cluster_rows or cluster_cols:
        if cluster_rows:
            row_linkage = linkage(data.values, method='average')
            row_order = leaves_list(row_linkage)
            data = data.iloc[row_order, :]
            
        if cluster_cols:
            col_linkage = linkage(data.T.values, method='average')
            col_order = leaves_list(col_linkage)
            data = data.iloc[:, col_order]
    
    # Mask diagonal entries based on matching index and column names
    # This works correctly even after clustering (unlike np.fill_diagonal)
    if mask_diagonal:
        for idx in data.index:
            if idx in data.columns:
                data.loc[idx, idx] = np.nan

    # Make values below `cmin` NaN
    if cmin is not None:
        data = data.mask(data.abs() < cmin)

    # Step 1 & 3: Subset and insert gaps based on panel_groups
    if panel_groups is not None:
        if isinstance(panel_groups, tuple):
            row_groups, col_groups = panel_groups, panel_groups
        elif isinstance(panel_groups, list) and len(panel_groups) == 2:
            row_groups, col_groups = panel_groups
            if row_groups is None:
                row_groups = [data.index.tolist()]
            if col_groups is None:
                col_groups = [data.columns.tolist()]
        else:
            raise ValueError("panel_groups must be a tuple or list of two lists (row_groups, col_groups)")
        
        row_groups = [
            [r for r in group if r in data.index]
            for group in row_groups
        ]
        row_groups = [group for group in row_groups if len(group)>0]
        
        col_groups = [
            [c for c in group if c in data.columns]
            for group in col_groups
        ]
        col_groups = [group for group in col_groups if len(group)>0]

        # Subset data to only include specified rows/columns (in order)
        all_rows = [r for group in row_groups for r in group]
        all_cols = [c for group in col_groups for c in group]
        data = data.loc[all_rows, all_cols]
        
        # Insert empty columns between column groups
        col_positions = []
        current_pos = 0
        for i, group in enumerate(col_groups):
            current_pos += len(group)
            if i < len(col_groups) - 1:  # Don't add gap after last group
                col_positions.append(current_pos)
                data.insert(current_pos, f' ' * (i + 1), np.nan)  # Unique name for each gap
                current_pos += 1
    else:
        row_groups = [data.index.tolist()]
        
    # Step 4: Determine subplot layout and height ratios
    n_panels = len(row_groups)
    if n_panels > 1:
        heights = [len(group) for group in row_groups]
        heights[-1] += 2  # Add extra space to last panel for colorbar
        
        # Calculate hspace to match horizontal gap width
        # Horizontal gap = 1 column width relative to total width
        # Vertical gap should be ~1 row height relative to average panel height
        avg_height = sum(heights) / len(heights)
        hspace_value = 1.0 / avg_height  # Gap of ~1 row height between panels
        for i in range(1, len(heights) - 1):
            heights[i] += hspace_value
        
        fig, axes = plt.subplots(
            n_panels, 1, 
            figsize=(data.shape[1] * figsize_multiplier[0], data.shape[0] * figsize_multiplier[1]),
            height_ratios=heights,
            gridspec_kw={'hspace': hspace_value},
            dpi=300
        )
        if not isinstance(axes, np.ndarray):
            axes = [axes]
    else:
        fig, ax = plt.subplots(
            1, 1,
            figsize=(data.shape[1] * figsize_multiplier[0], data.shape[0] * figsize_multiplier[1]),
            dpi=300
        )
        axes = [ax]
    
    # Step 5: Plot heatmaps for each panel
    for i, (ax, row_group) in enumerate(zip(axes, row_groups)):
        # Select data for this panel
        panel_data = data.loc[row_group, :]
        
        cbar_kws_ = {'orientation': 'horizontal', 'shrink': 0.6, 'pad': 0.05}
        if clegend_title and i == len(axes) - 1:
            cbar_kws_['label'] = clegend_title
        
        # Create heatmap
        sns.heatmap(
            panel_data,
            square=False,
            annot=annot,
            fmt=fmt,
            annot_kws={'fontsize': annot_fontsize},
            cmap=cmap,
            vmin=clims[0],
            vmax=clims[1],
            cbar=(i == len(axes) - 1),  # Only show colorbar on last panel
            cbar_kws=cbar_kws_,
            linewidths=linewidths,
            linecolor=linecolor,
            ax=ax
        )
        
        ax.collections[0].cmap.set_bad(na_color)

        # Set grey background with white grid lines
        ax.set_facecolor("#f0f0f0")
        ax.grid(False)
        ax.set_axisbelow(True)
        
        # Overprint gap columns with wide white vertical lines
        for j, col in enumerate(panel_data.columns):
            if str(col).strip() == '':  # Gap column (named with spaces)
                ax.axvline(j + 0.5, color='white', linewidth=20, zorder=10)
        
        # Configure axes
        if i == 0:
            # First panel: x-labels on top
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=xlabel_rotation, ha=xlabel_ha)
        else:
            # Other panels: no x-labels
            ax.set_xticklabels([])
        
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    

    return fig, axes


def compute_attribute_associations(df: pd.DataFrame, attrs_a: list, attrs_b: list = None, 
                                  log_base=np.e, eps=1e-12) -> dict:
    """
    Compute co-occurrence counts and association measures between attribute sets.
    
    If attrs_b is None, computes associations within attrs_a (useful for within-dimension analysis).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing attribute columns
    attrs_a : list
        First set of attribute column names
    attrs_b : list, optional
        Second set of attribute column names. If None, uses attrs_a.
    log_base : float
        Base for logarithm (np.e for natural log, 2 for bits, 10 for common log)
    eps : float
        Small value to avoid log(0)
    
    Returns
    -------
    dict with DataFrames:
        - cooc_counts: Co-occurrence counts (|A| x |B|)
        - pmi: Pointwise Mutual Information
        - ppmi: Positive PMI (max(PMI, 0))
        - npmi: Normalized PMI (range [-1, 1])
        - p_b_given_a: Conditional probability P(B|A)
        - p_a_given_b: Conditional probability P(A|B)
        - marginals: dict with counts and probabilities for each label set
    """
    if attrs_b is None:
        attrs_b = attrs_a
    
    # Binarize and convert to numpy arrays
    A = df[attrs_a].apply(lambda col: binarize_column(col)).astype(int).to_numpy()
    B = df[attrs_b].apply(lambda col: binarize_column(col)).astype(int).to_numpy()
    N = A.shape[0]
    
    # Joint and marginal counts
    C = A.T @ B                              # joint counts: |A| x |B|
    C_A = A.sum(axis=0).astype(float)        # counts for A labels
    C_B = B.sum(axis=0).astype(float)        # counts for B labels
    
    # Probabilities
    Pab = C / max(N, 1)
    Pa = C_A / max(N, 1)
    Pb = C_B / max(N, 1)
    
    # Choose log base
    def _log(x):
        if log_base == 2:
            return np.log2(x)
        elif log_base == 10:
            return np.log10(x)
        return np.log(x)
    
    # PMI: log( P(A,B) / (P(A)P(B)) )
    # Measures how much more (or less) often two attributes co-occur than expected under independence
    PMI = _log((Pab + eps) / ((Pa[:, None] + eps) * (Pb[None, :] + eps)))
    
    # PPMI: max(PMI, 0)
    # Only keeps positive associations (co-occurrence above chance)
    PPMI = np.maximum(PMI, 0.0)
    
    # nPMI: PMI / (-log P(A,B))
    # Normalized to [-1, 1] range for easier interpretation
    nPMI = PMI / (-_log(Pab + eps))
    
    # Conditional probabilities (more interpretable than PMI)
    P_b_given_a = (C + eps) / (C_A[:, None] + eps)   # P(B|A)
    P_a_given_b = (C + eps) / (C_B[None, :] + eps)   # P(A|B)
    
    # Create DataFrames with proper labels
    idx = pd.Index(attrs_a, name="attr_a")
    cols = pd.Index(attrs_b, name="attr_b")
    
    return {
        "cooc_counts": pd.DataFrame(C, index=idx, columns=cols),
        "pmi": pd.DataFrame(PMI, index=idx, columns=cols),
        "ppmi": pd.DataFrame(PPMI, index=idx, columns=cols),
        "npmi": pd.DataFrame(nPMI, index=idx, columns=cols),
        "p_b_given_a": pd.DataFrame(P_b_given_a, index=idx, columns=cols),
        "p_a_given_b": pd.DataFrame(P_a_given_b, index=idx, columns=cols),
        "marginals": {
            "N": N,
            "count_a": pd.Series(C_A, index=idx),
            "count_b": pd.Series(C_B, index=cols),
            "p_a": pd.Series(Pa, index=idx),
            "p_b": pd.Series(Pb, index=cols),
        }
    }

def label_assoc_matrices(df, label_cols_a, label_cols_b, log_base=np.e, eps=1e-12):
    """
    Compute co-occurrence counts and association measures between two multilabel schemes.

    Returns dict with:
      - cooc_counts: DataFrame (|A| x |B|)
      - pmi:         DataFrame
      - ppmi:        DataFrame
      - npmi:        DataFrame (in [-1, 1])
      - p_b_given_a: DataFrame of P(B_j | A_i)
      - p_a_given_b: DataFrame of P(A_i | B_j)
      - marginals:   dict with Series for counts/probs
    """
    # Indicator matrices
    A = df[label_cols_a].astype(int).to_numpy()
    B = df[label_cols_b].astype(int).to_numpy()
    N = A.shape[0]

    # Counts
    C = A.T @ B                              # joint counts: |A| x |B|
    C_A = A.sum(axis=0).astype(float)        # counts for A labels: |A|
    C_B = B.sum(axis=0).astype(float)        # counts for B labels: |B|

    # Probabilities
    Pab = C / max(N, 1)
    Pa = C_A / max(N, 1)
    Pb = C_B / max(N, 1)

    # Use chosen log base
    def _log(x):
        if log_base == 2:
            return np.log2(x)
        if log_base == 10:
            return np.log10(x)
        return np.log(x)

    # PMI: log( P(A,B) / (P(A)P(B)) )
    PMI = _log((Pab + eps) / ((Pa[:, None] + eps) * (Pb[None, :] + eps)))

    # PPMI: max(PMI, 0)
    PPMI = np.maximum(PMI, 0.0)

    # nPMI: PMI / (-log P(A,B))
    nPMI = PMI / (-_log(Pab + eps))

    # Conditional probabilities (interpretable)
    P_b_given_a = (C + eps) / (C_A[:, None] + eps)   # P(B|A)
    P_a_given_b = (C + eps) / (C_B[None, :] + eps)   # P(A|B)

    # Wrap into DataFrames
    idx = pd.Index(label_cols_a, name="A_label")
    cols = pd.Index(label_cols_b, name="B_label")

    out = {
        "cooc_counts": pd.DataFrame(C, index=idx, columns=cols),
        "pmi":         pd.DataFrame(PMI, index=idx, columns=cols),
        "ppmi":        pd.DataFrame(PPMI, index=idx, columns=cols),
        "npmi":        pd.DataFrame(nPMI, index=idx, columns=cols),
        "p_b_given_a": pd.DataFrame(P_b_given_a, index=idx, columns=cols),
        "p_a_given_b": pd.DataFrame(P_a_given_b, index=idx, columns=cols),
        "marginals": {
            "N": N,
            "count_a": pd.Series(C_A, index=idx),
            "count_b": pd.Series(C_B, index=cols),
            "p_a": pd.Series(Pa, index=idx),
            "p_b": pd.Series(Pb, index=cols),
        }
    }
    return out