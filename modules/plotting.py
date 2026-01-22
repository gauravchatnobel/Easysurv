import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import numpy as np

def add_at_risk_counts(fitters, ax=None, y_shift=-0.25, colors=None, labels=None, fontsize=10):
    """
    Add a table of at-risk counts below the plot.
    Re-implemented using ax.text for perfect alignment with X-axis ticks.
    """
    if ax is None:
        ax = plt.gca()
    
    # Get ticks from the plot
    ticks = ax.get_xticks()
    # Filter ticks that make sense AND are within the current view limits
    view_min, view_max = ax.get_xlim()
    valid_ticks = [t for t in ticks if view_min <= t <= view_max]
    
    # Configuration for layout
    row_height = 0.05
    start_y = y_shift
    
    # Use a blended transform: X is data coords (so it matches ticks), Y is axes coords (so it stays absolute relative to plot bottom)
    # We actually need two transforms: 
    # 1. For data numbers: x=data, y=axes
    # 2. For row labels: x=axes (negative), y=axes
    trans_data_axes = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    
    for i, fitter in enumerate(fitters):
        y_pos = start_y - (i * row_height)
        
        # 1. Plot Row Label (Left of Y-axis)
        # Use provided custom label if available, else fitter label
        lbl = labels[i] if labels and i < len(labels) else fitter._label
        
        # Color resolution
        color = 'black'
        if colors and i < len(colors):
            color = colors[i]
            
        ax.text(-0.03, y_pos, lbl, transform=ax.transAxes, 
                ha='right', va='center', weight='bold', color=color, fontsize=fontsize)
        
        # 2. Plot Counts at each tick
        for t in valid_ticks:
            # Calculate at risk
            if t in fitter.event_table.index:
                val = fitter.event_table.loc[t, 'at_risk']
            else:
                sliced = fitter.event_table.loc[:t]
                if sliced.empty:
                    val = fitter.event_table['at_risk'].iloc[0]
                else:
                    last_row = sliced.iloc[-1]
                    val = last_row['at_risk'] - last_row['removed']
            
            if isinstance(val, (pd.Series, np.ndarray, list)):
                try:
                    val = val.item()
                except:
                    pass
            val = int(val)
            
            # Plot the number
            ax.text(t, y_pos, str(val), transform=trans_data_axes, 
                    ha='center', va='center', color=color, fontsize=fontsize, weight='bold')
