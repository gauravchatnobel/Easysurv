
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
import numpy as np
import numpy as np
import io
try:
    import seaborn as sns
except ImportError:
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        import seaborn as sns
    except Exception as e:
        # Fallback if pip install fails, though heatmap will fail later.
        sns = None

@st.cache_data
def compute_fine_gray_weights(df, time_col, event_col, event_of_interest=1):
    """
    Prepares a dataset for Fine-Gray regression using Inverse Probability of Censoring Weighting (IPCW).
    Ref: Fine JP, Gray RJ. A proportional hazards model for the subdistribution of a competing risk. 
    J Am Stat Assoc. 1999;94(446):496–509.
    """
    df = df.copy()
    df = df.sort_values(time_col)
    
    # 1. Estimate Censoring Distribution G(t)
    censoring_df = df.copy()
    censoring_df['cens_event'] = (censoring_df[event_col] == 0).astype(int)
    
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(censoring_df[time_col], censoring_df['cens_event'])
    
    def get_G(t):
        probs = kmf_c.survival_function_at_times(t).values
        # Ensure we return a scalar float
        if np.isscalar(probs):
             return float(probs)
        else:
             return float(probs.item()) if probs.size == 1 else float(probs[0])

    # 2. Identify Event Times of Interest
    event_times = df[df[event_col] == event_of_interest][time_col].unique()
    event_times = np.sort(event_times)
    
    # 3. Build Expanded Dataset
    new_rows = []
    if 'id' not in df.columns:
        df['id'] = range(len(df))
    
    for _, row in df.iterrows():
        t = row[time_col]
        e = row[event_col]
        pid = row['id']
        
        # Case A: Event or Censored - contribute normally
        if e == event_of_interest or e == 0:
            new_rows.append({
                'id': pid, 'start': 0, 'stop': t,
                'status': 1 if e == event_of_interest else 0,
                'weight': 1.0,
                **{c: row[c] for c in df.columns if c not in [time_col, event_col, 'id']}
            })
            
        # Case B: Competing Event - remain in risk set with decaying weights
        elif e != event_of_interest and e > 0:
            # Interval [0, Ti]
            new_rows.append({
                'id': pid, 'start': 0, 'stop': t,
                'status': 0, 'weight': 1.0,
                **{c: row[c] for c in df.columns if c not in [time_col, event_col, 'id']}
            })
            
            # Extension [Ti, tk]
            relevant_times = event_times[event_times > t]
            if len(relevant_times) > 0:
                G_Ti = max(get_G(t), 1e-5)
                current_start = t
                for rt in relevant_times:
                    weight = get_G(rt) / G_Ti
                    new_rows.append({
                        'id': pid, 'start': current_start, 'stop': rt,
                        'status': 0, 'weight': weight,
                        **{c: row[c] for c in df.columns if c not in [time_col, event_col, 'id']}
                    })
                    current_start = rt
                    if weight < 1e-4: break

    return pd.DataFrame(new_rows)

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
    import matplotlib.transforms as mtransforms
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
    
    # Adjust layout to make room for the table and labels
    # Table height approx = row_height * len(fitters)
    # Plus some initial offset absolute value
    # But usually plt.subplots_adjust or bbox_inches='tight' in savefig handles this.
    # We will assume the user might need to adjust plot margins manually if labels are very long.



st.set_page_config(page_title="Survival Analysis Tool", layout="wide")

st.title("EASYSURV: Interactive Survival Analysis Tool")

# Sidebar - Configuration
st.sidebar.header("Data Upload & Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Clinical Data (CSV/Excel)", type=["csv", "xlsx"])

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        return None

df = None

if uploaded_file:
    df = load_data(uploaded_file)
else:
    # --- LANDING PAGE ---
    st.markdown("""
    <style>
    .hero-box {
        padding: 2rem;
        background-color: #f0f2f6; 
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hero-box">', unsafe_allow_html=True)
    st.markdown("## 👋 Welcome to EasySurv")
    st.markdown("### Survival Analysis for everyone")
    st.markdown("Perform publication-quality Kaplan-Meier, Cox Regression, and Competing Risks analysis in seconds without writing a single line of code.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📊 Interactive Plots**\n\nCreate publication-quality curves with aligned risk tables and custom themes.")
    with col2:
        st.success("**🤖 AI Narrator**\n\nGet instant natural language summaries of your P-values and Hazard Ratios.")
    with col3:
        st.warning("**🧬 Biomarker Optimum Threshold**\n\nAutomatically find optimal cutoffs and visualize correlations.")

    st.divider()
    
    col_demo, col_blank = st.columns([1, 4])
    with col_demo:
         if st.button("🚀 Load Demo Data", type="primary", use_container_width=True):
             # Load directly without cache to identify file changes
             try:
                 # Check multiple locations
                 import os
                 possible_paths = ["dummy_clinical_data.csv", "survival_analysis/dummy_clinical_data.csv"]
                 found_path = None
                 for p in possible_paths:
                     if os.path.exists(p):
                         found_path = p
                         break
                 
                 if found_path:
                     df = pd.read_csv(found_path) # Direct read, no cache
                 else:
                     st.error("Demo file not found.")
                     
             except Exception as e:
                 st.error(f"Error loading demo: {e}")
             
             if df is not None:
                 st.session_state['demo_loaded'] = True
                 st.rerun()

    # Check session state for demo persistence
    if st.session_state.get('demo_loaded', False):
        try:
             import os
             possible_paths = ["dummy_clinical_data.csv", "survival_analysis/dummy_clinical_data.csv"]
             for p in possible_paths:
                 if os.path.exists(p):
                     df = pd.read_csv(p) # Direct read
                     break
        except:
             pass

if df is not None:


        


    # --- SESSION STATE & CUSTOM VARIABLES ---
    if 'custom_cutoffs' not in st.session_state:
        st.session_state.custom_cutoffs = []
        
    # Apply valid custom cutoffs to df
    if st.session_state.custom_cutoffs:
        for cutoff_def in st.session_state.custom_cutoffs:
             src = cutoff_def['source']
             val = cutoff_def['value']
             name = cutoff_def['name']
             
             if src in df.columns:
                 # Create categorical column
                 # "High" if > val, "Low" if <= val
                 df[name] = np.where(df[src] > val, f"High (> {val:.2f})", f"Low (<= {val:.2f})")
    
    # NEW: Apply custom variable combinations (Interactions)
    if 'custom_combinations' not in st.session_state:
        st.session_state.custom_combinations = []
        
    if st.session_state.custom_combinations:
        for combo_def in st.session_state.custom_combinations:
            if 'type' not in combo_def or combo_def['type'] == 'interaction':
                var1 = combo_def['var1']
                var2 = combo_def['var2']
                new_name = combo_def['name']
                
                if var1 in df.columns and var2 in df.columns:
                     # Concatenate values with a separator
                     df[new_name] = df[var1].astype(str) + " / " + df[var2].astype(str)
                     
                     # NEW: Set to NaN if either source is NaN (don't create string "nan / ...")
                     mask_nan = df[var1].isna() | df[var2].isna()
                     df.loc[mask_nan, new_name] = np.nan
            
            elif combo_def['type'] == 'boolean':
                try:
                    # Boolean Logic
                    c1 = combo_def['c1']
                    logic = combo_def['logic']
                    c2 = combo_def['c2']
                    name = combo_def['name']
                    
                    # Helper to get mask
                    def get_mask(col, op, val):
                        if op == ">": return df[col] > val
                        if op == "<": return df[col] < val
                        if op == ">=": return df[col] >= val
                        if op == "<=": return df[col] <= val
                        if op == "==": return df[col] == val
                        if op == "!=": return df[col] != val
                        return pd.Series([False]*len(df))
                    
                    mask1 = get_mask(c1['var'], c1['op'], c1['val'])
                    
                    if c2:
                        mask2 = get_mask(c2['var'], c2['op'], c2['val'])
                        if logic == "AND":
                            final_mask = mask1 & mask2
                        else:
                            final_mask = mask1 | mask2
                    else:
                        final_mask = mask1
                        
                    df[name] = np.where(final_mask, "Target Group", "Other")
                except Exception as e:
                    st.error(f"Failed to create boolean variable {combo_def['name']}: {e}")
    
    # --- DATA FILTRATION ---
    # --- DATA FILTRATION ---
    with st.expander("🔍 Step 1: Filter Data (Optional)", expanded=True):
        st.write("Select subset of data based on categorical variables (e.g., specific ELN Risk groups).")
        all_cols_filter = df.columns.tolist()
        filter_cols = st.multiselect("Select Columns to Filter By", all_cols_filter)
        
        if filter_cols:
            df_filtered = df.copy()
            for col in filter_cols:
                # Check for numeric columns (to show slider vs multiselect)
                if pd.api.types.is_numeric_dtype(df[col]) and len(df[col].unique()) > 10:
                     min_val = float(df[col].min())
                     max_val = float(df[col].max())
                     step = (max_val - min_val) / 100.0 if max_val != min_val else 0.1
                     
                     st.markdown(f"**Filter '{col}' (Numeric Range)**")
                     rng = st.slider(f"Select Range: {col}", min_val, max_val, (min_val, max_val), step=step, key=f"filt_{col}")
                     
                     # Apply Filter
                     df_filtered = df_filtered[df_filtered[col].between(rng[0], rng[1])]
                else:
                    # Categorical / Low-cardinality Numeric
                    unique_vals = sorted(df[col].dropna().unique())
                    selected_vals = st.multiselect(f"Select values to KEEP for '{col}'", unique_vals, default=unique_vals, key=f"filt_{col}")
                    
                    # Apply filter
                    if selected_vals:
                        df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]
            
            # Show stats
            n_before = len(df)
            n_after = len(df_filtered)
            st.metric("Rows Remaining", f"{n_after} / {n_before}", delta=n_after-n_before)
            
            # Update the main dataframe
            df = df_filtered

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Column Selection
    columns = df.columns.tolist()
    
    st.sidebar.subheader("Variable Selection")
    
    # Time and Event columns
    # Time and Event columns
    # Smart Defaults for Demo
    default_time_idx = 0
    default_event_idx = 0
    default_group_idx = 0
    
    if st.session_state.get('demo_loaded', False):
        if "OS_Months" in columns:
            default_time_idx = columns.index("OS_Months")
        if "Event_Occurred" in columns:
            default_event_idx = columns.index("Event_Occurred")
        if "Treatment_Arm" in columns:
            # +1 because "None" is at index 0
            default_group_idx = columns.index("Treatment_Arm") + 1
    else:
        # Standard intelligent defaults
        if "OS_Days" in columns: default_time_idx = columns.index("OS_Days")
        elif "Time" in columns: default_time_idx = columns.index("Time")
        
        if "OS_Status" in columns: default_event_idx = columns.index("OS_Status")
        elif "Status" in columns: default_event_idx = columns.index("Status")
        
        if "MRD_Status" in columns: default_group_idx = columns.index("MRD_Status") + 1

    time_col = st.sidebar.selectbox("Time Column (Duration)", columns, index=default_time_idx)
    event_col = st.sidebar.selectbox("Event Column (Status: 1=Event, 0=Censored)", columns, index=default_event_idx)
    
    # Grouping Variable
    group_col = st.sidebar.selectbox("Grouping Variable (e.g., MRD Status)", ["None"] + columns, index=default_group_idx)

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("Configuration")
    
    # 1. Global Aesthetics
    st.sidebar.subheader("Global Theme & Typography")
    font_options = ["sans-serif", "serif", "monospace", "Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana", "Comic Sans MS"]
    selected_font = st.sidebar.selectbox("Font Family", font_options, index=0)
    plt.rcParams['font.family'] = selected_font
    
    # Title customizations (Shared)
    title_fontsize = st.sidebar.slider("Title Font Size", 10, 30, 20)
    title_bold = st.sidebar.checkbox("Bold Title", value=True)
    title_fontweight = 'bold' if title_bold else 'normal'

    p_val_fontsize = 12 # Default

    axes_fontsize = st.sidebar.number_input("Axes/Tick Font Size", min_value=6, value=12)
    legend_fontsize = st.sidebar.number_input("Legend Font Size", min_value=6, value=10)
    line_width = st.sidebar.slider("Line Width", 0.5, 5.0, 1.5)
    
    # Global Plot Configuration (Elements affecting all plots)
    st.sidebar.subheader("Global Plot Configuration")
    show_risk_table = st.sidebar.checkbox("Show At-Risk Table", value=True)
    table_height = st.sidebar.slider("Table Offset", -0.5, -0.1, -0.25, 0.05) if show_risk_table else -0.25
    show_censored = st.sidebar.checkbox("Show Censored Ticks", value=True)
    show_ci = st.sidebar.checkbox("Show 95% CI", value=True)
    
    # 2. Main Plot Settings (Kaplan-Meier)
    with st.sidebar.expander("Main Plot Settings (KM)", expanded=False):
        st.markdown("### Layout & Axes")
        main_title = st.text_input("Main Plot Title", value="Survival")
        x_label = st.text_input("X-Axis Label", value="Time (Months)")
        y_label = st.text_input("Y-Axis Label", value="Survival Probability")
        
        col1, col2 = st.columns(2)
        with col1:
            tick_interval = st.number_input("X-Tick Step", value=12.0)
            y_min = st.number_input("Y Min", value=0.0, step=0.1)
        with col2:
            y_tick_interval = st.number_input("Y-Tick Step", value=0.1)
            y_max = st.number_input("Y Max", value=1.0, step=0.1)
            
        plot_height = st.slider("Plot Height", 4, 12, 6)
        plot_width = st.slider("Plot Width", 6, 15, 10)
        
        # Elements (Moved to Global)
        
        st.markdown("### Legend")
        show_legend_main = st.checkbox("Show Legend (Main)", value=True)
        show_legend_box_main = st.checkbox("Box Legend (Main)", value=True)
        leg_x_main = st.slider("Legend X (Main)", 0.0, 1.0, 0.8)
        leg_y_main = st.slider("Legend Y (Main)", 0.0, 1.0, 0.9)
        
        st.markdown("### P-value")
        show_p_val_plot = st.checkbox("Show P-value (Main)", value=False)
        show_p_val_box_main = st.checkbox("Box P-value (Main)", value=True)
        pval_x_main = st.slider("P-val X (Main)", 0.0, 1.0, 0.95)
        pval_y_main = st.slider("P-val Y (Main)", 0.0, 1.0, 0.05)

        st.markdown("### Free Text Annotation")
        main_free_text = st.text_input("Add Text (Main)", value="", placeholder="e.g. HR=0.45")
        main_text_x = st.slider("Text X (Main)", 0.0, 1.0, 0.5)
        main_text_y = st.slider("Text Y (Main)", 0.0, 1.0, 0.5)
        main_text_size = st.number_input("Text Size (Main)", min_value=6, value=12)
        main_text_box = st.checkbox("Box Text (Main)", value=False)

    # 3. CIF Plot Settings
    with st.sidebar.expander("CIF Plot Settings (Competing Risks)", expanded=False):
        st.markdown("### Layout & Axes")
        cif_title = st.text_input("CIF Plot Title", value="Cumulative Incidence")
        cif_y_label = st.text_input("CIF Y-Label", value="Cumulative Incidence Probability")
        
        col3, col4 = st.columns(2)
        with col3:
            cif_y_min = st.number_input("CIF Y Min", value=0.0, step=0.1)
            cif_y_tick_interval = st.number_input("CIF Y-Tick Step", value=0.1)
        with col4:
            cif_y_max = st.number_input("CIF Y Max", value=1.05, step=0.1)
        
        st.markdown("### Legend")
        show_legend_cif = st.checkbox("Show Legend (CIF)", value=True)
        show_legend_box_cif = st.checkbox("Box Legend (CIF)", value=True)
        leg_x_cif = st.slider("Legend X (CIF)", 0.0, 1.0, 0.8)
        leg_y_cif = st.slider("Legend Y (CIF)", 0.0, 1.0, 0.8)
        
        st.markdown("### P-value")
        show_p_val_plot_cif = st.checkbox("Show P-value (CIF)", value=False)
        show_p_val_box_cif = st.checkbox("Box P-value (CIF)", value=True)
        pval_x_cif = st.slider("P-val X (CIF)", 0.0, 1.0, 0.95)
        pval_y_cif = st.slider("P-val Y (CIF)", 0.0, 1.0, 0.2)
        
        st.markdown("### Free Text Annotation")
        cif_free_text = st.text_input("Add Text (CIF)", value="", placeholder="e.g. p=0.003")
        cif_text_x = st.slider("Text X (CIF)", 0.0, 1.0, 0.5)
        cif_text_y = st.slider("Text Y (CIF)", 0.0, 1.0, 0.5)
        cif_text_size = st.number_input("Text Size (CIF)", min_value=6, value=12)
        cif_text_box = st.checkbox("Box Text (CIF)", value=False)
        
    # Theme Selection (moved to bottom or keep global)
    st.sidebar.subheader("Color Theme")

    # Theme Selection
    st.sidebar.subheader("Aesthetics & Themes")
    
    # Define Palettes
    # Define Palettes (Source: ggsci)
    journal_themes = {
        "Nature (NPG)": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
        "Science (AAAS)": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021", "#5F559B", "#A20056", "#808180", "#1B1919"],
        "JCO": ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67", "#8F7700", "#3B3B3B", "#A73030", "#4A6990"],
        "Lancet": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6", "#1B1919"],
        "NEJM": ["#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91", "#EE4C97"],
        "Blood": ["#AA0000", "#E00000", "#8B0000", "#FF0000", "#B22222", "#FF69B4", "#800000"], 
        "Leukemia": ["#377EB8", "#E41A1C", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF"], 
        "Jama": ["#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599", "#80796B"],
    }
    
    fun_themes = {
        "Sci-Fi Duo": ["#A6EEE6", "#F0F035", "#44281D", "#E4A71B", "#8BCF21", "#FBFBFB"],
        "Hollywood Equine": ["#2C3E50", "#D35400", "#2980B9", "#C0392B", "#bdc3c7", "#F39C12"],
        "Prehistoric One": ["#5D4037", "#D84315", "#388E3C", "#FBC02D", "#455A64", "#212121"],
        "Alien Flora": ["#F4EBD0", "#D66853", "#3C505D", "#7D9D9C", "#212D40", "#A8C686"],
        "Neon Acid": ["#FF00FF", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF0000"],
        "Cyber Grid": ["#2F5061", "#4291C7", "#D57FBE", "#E45D5C", "#FFAE91", "#F9DB57", "#FFFFD0"],
        "Dream Layers": ["#EFAC3D", "#842650", "#69932B", "#5F2D7B", "#5C5D5E"],
        "Deep Space": ["#0B3D91", "#F2C45A", "#465362", "#A2BCE0", "#1E1E24"],
        "Office Separation": ["#F0F6F7", "#89AEC8", "#7B6727", "#4E452A", "#002C55"],
        "Sudden Departure": ["#F2A78C", "#F2DCBB", "#B4B4BB", "#77708C", "#C997A2"],
        "Family Empire": ["#1C2541", "#3A506B", "#5BC0BE", "#6FFFE9", "#0B132B"],
        "Practice Run": ["#e0e1dd", "#778da9", "#415a77", "#1b263b", "#0d1b2a"],
        "Exclusion Zone": ["#D8FF00", "#E0FF33", "#333333", "#555555", "#D65050"],
        "Seven Kingdoms": ["#808080", "#FFD700", "#B22222", "#000000", "#228B22"],
        "Indian Train Journey": ["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"],
        "Grand Hotel": ["#F1BB7B", "#FD6467", "#5B1A18", "#D67236"],
        "Gotham Night": ["#0C2340", "#282D3C", "#808080", "#000000", "#710193", "#32CD32", "#B3B3B3"],
        "Coal Town Saga": ["#212121", "#B71C1C", "#FFC107", "#5D4037", "#00C853", "#E65100"],
        "Feudal Colors": ["#FFD700", "#DC143C", "#0000CD", "#DAA520", "#000000", "#FFFFFF"],
        "Classic Monochrome": ["#1A1A1A", "#4D4D4D", "#808080", "#B3B3B3", "#E6E6E6", "#F0EAD6"], 
        "Anime Fantasy": ["#8CBF88", "#E53935", "#607D8B", "#FFA500", "#D2E3EF", "#FF6347"],
        
        # Literary Themes
        "Saint Petersburg 1866": ["#3E2723", "#BF360C", "#F9A825", "#424242", "#ECEFF1"], # Crime & Punishment: Squalor, blood, feverish yellow, stone grey, pale sky
        "The Law Clerk": ["#263238", "#546E7A", "#78909C", "#D7CCC8", "#8D6E63", "#212121"], # The Trial: Bureaucracy, paper, ink, oppressive grey
        "Stream of Life": ["#D81B60", "#F48FB1", "#4A148C", "#FFF176", "#00BCD4"], # Agua Viva/G.H.: Visceral pink, organic purple, blinding yellow, fluid blue
        "Cosmic Ocean": ["#311B92", "#7C4DFF", "#00E676", "#3E2723", "#FF6F00"], # Solaris/Stalker: Deep space purple, irradiated green, rust, amber
        "The Playwright": ["#607D8B", "#8D6E63", "#CFD8DC", "#A1887F", "#546E7A"], # Chekhov: Muted, melancholic, realistic earth tones
    }
    
    all_themes = {**journal_themes, **fun_themes}
    theme_names = ["Default"] + list(journal_themes.keys()) + list(fun_themes.keys()) + ["Custom"]
    
    selected_theme = st.sidebar.selectbox("Choose Theme", theme_names)
    
    # --- DOWNLOAD MODIFIED DATA (At bottom of sidebar) ---
    st.sidebar.divider()
    # Use df (which has new vars added at the top) or df_clean (which filters NaNs). 
    # Usually users want the full dataset with new variables, so 'df' is better, 
    # but 'df_filtered' if they want the filtered view. Let's give them the full 'df' with enhancements.
    if df is not None:
        csv_buffer = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="💾 Download Enhanced Data (CSV)",
            data=csv_buffer,
            file_name="easysurv_enhanced_data.csv",
            mime="text/csv",
            help="Download the dataset including all new variables created in this session."
        )
    
    # --- REPORT GENERATOR ---
    st.sidebar.divider()
    st.sidebar.subheader("📄 Report Generator")
    if st.sidebar.button("Generate HTML Report"):
        import base64
        
        def fig_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        # 1. Dataset stats
        n_rows, n_cols = df.shape if df is not None else (0,0)
        cols_list = ", ".join(df.columns) if df is not None else "None"
        
        # 2. Plots
        img_km = ""
        if 'report_fig_km' in st.session_state:
             img_km = f'<img src="data:image/png;base64,{fig_to_base64(st.session_state["report_fig_km"])}" style="width:100%">'
        else:
             img_km = "<p><em>No Univariable Plot generated yet.</em></p>"
             
        img_forest = ""
        if 'report_fig_forest' in st.session_state:
             img_forest = f'<img src="data:image/png;base64,{fig_to_base64(st.session_state["report_fig_forest"])}" style="width:100%">'
        else:
             img_forest = "<p><em>No Multivariable Forest Plot generated yet.</em></p>"

        # 3. HTML Template
        html_report = f"""
        <html>
        <head>
            <title>EasySurv Analysis Report</title>
            <style>
                body {{ font-family: sans-serif; max_width: 800px; margin: auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2c3e50; margin-top: 30px; border-bottom: 1px solid #ddd; }}
                .section {{ margin-bottom: 40px; }}
                .meta {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>Survival Analysis Report</h1>
            <p class="meta">Generated by EasySurv</p>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <p><strong>Rows:</strong> {n_rows} | <strong>Columns:</strong> {n_cols}</p>
                <p><strong>Variables:</strong> {cols_list}</p>
            </div>
            
            <div class="section">
                <h2>1. Univariable Analysis</h2>
                {img_km}
            </div>
            
            <div class="section">
                <h2>2. Multivariable Analysis</h2>
                {img_forest}
            </div>
            
            <div class="section">
                <h2>Notes</h2>
                <p>This report contains snapshots of the latest plots generated in your session.</p>
            </div>
        </body>
        </html>
        """
        
        # Download Button via a trick or standard st.download_button
        # But we are inside a button... Nested buttons don't work well in Streamlit.
        # However, saving it to session state and creating a download button *outside* is better.
        # But st.download_button works if we just render it now.
        
        b64_html = base64.b64encode(html_report.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64_html}" download="easysurv_report.html" target="_blank" style="text-decoration:none; color:white; background-color:#ff4b4b; padding:8px 16px; border-radius:5px;">⬇️ Download Report (HTML)</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.sidebar.success("Report Ready! Click above.")
    
    custom_colors = {}
    # Data Cleaning for NaNs
    df_clean = None
    if time_col and event_col:
        cols_to_check = [time_col, event_col]
        if group_col != "None":
            cols_to_check.append(group_col)
            
        original_count = len(df)
        df_clean = df.dropna(subset=cols_to_check).copy()
        dropped_count = original_count - len(df_clean)
        
        # Ensure correct types
        try:
            df_clean[time_col] = pd.to_numeric(df_clean[time_col], errors='coerce')
            df_clean[event_col] = pd.to_numeric(df_clean[event_col], errors='coerce')
            # Check for any new NaNs created by coercion
            if df_clean[time_col].isna().sum() > 0 or df_clean[event_col].isna().sum() > 0:
                 st.error("Error: Non-numeric data found in Time or Event columns.")
                 st.stop()
        except Exception as e:
            st.error(f"Data type conversion error: {e}")
            st.stop()
            
    
    # Plot Background Color
    plot_bgcolor = st.sidebar.color_picker("Plot Background Color", "#FFFFFF")
    
    # Legend Renaming & Custom Colors
    group_labels = {}
    if group_col != "None" and df_clean is not None:
        with st.sidebar.expander("🎨 Custom Colors & Labels", expanded=(selected_theme == "Custom")):
            unique_groups = sorted(df_clean[group_col].dropna().unique())
            for grp in unique_groups:
                col1, col2 = st.columns([1, 1])
                with col1:
                    new_label = st.text_input(f"Label for {grp}", value=str(grp), key=f"label_{grp}")
                    group_labels[grp] = new_label
                with col2:
                    if selected_theme == "Custom":
                        color = st.color_picker(f"Color for {grp}", key=f"color_{grp}")
                        custom_colors[grp] = color

    # Analysis
    if time_col and event_col and df_clean is not None:
        st.divider()
        st.header("Survival Analysis")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Univariable (KM)", "Multivariable (Cox)", "Competing Risks (CIF)", "🧬 Biomarker Optimum Threshold", "🧪 Variable Generation", "🔥 Correlations"])

        with tab1:
        
            if dropped_count > 0:
                st.warning(f"⚠️ **Missing Values Detected**: Dropped {dropped_count} rows containing NaNs in the selected columns. Analysis based on {len(df_clean)} remaining rows.")
                with st.expander("See details of missing values"):
                    st.write("Number of missing values per column (in the original selection):")
                    # Calculate missing per column in the subset
                    missing_stats = df[cols_to_check].isna().sum()
                    st.dataframe(missing_stats[missing_stats > 0])
            
            # Calculate P-value for Plot if requested
            hr_text = ""
            p_value_text = None # Initialize to avoid NameError
            if show_p_val_plot and group_col != "None" and len(df_clean[group_col].unique()) >= 2:
                try:
                    cox_df = df_clean[[time_col, event_col, group_col]].dropna()
                    cox_data_encoded = pd.get_dummies(cox_df, columns=[group_col], drop_first=True)
                    cox_data_encoded.columns = [c.replace(' ', '_').replace('+', 'pos').replace('-', 'neg') for c in cox_data_encoded.columns]
                    cph_plot = CoxPHFitter()
                    cph_plot.fit(cox_data_encoded, duration_col=time_col, event_col=event_col)
                    if len(df_clean[group_col].unique()) == 2:
                         # For 2 groups, we might want to respect the reference group too, but usually it's just A vs B.
                         # We'll rely on the main CoxPH section below for detailed HR.
                         pass
                
                    # We will perform detailed stats below, just showing p-value on plot for now
                    res = multivariate_logrank_test(df_clean[time_col], df_clean[group_col], df_clean[event_col])
                    p_value_text = f"p = {res.p_value:.4f}"
                except:
                    pass

        
            # 1. Plotting
            # Force Font Update
            plt.rcParams['font.family'] = selected_font
            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        
            # Set Background Color
            fig.patch.set_facecolor(plot_bgcolor)
            ax.set_facecolor(plot_bgcolor)

            # Adjust Margins to accommodate Risk Table Labels (on the left) and Table (below)
            # We push the left margin to 0.2 (20%) to ensure labels like "Risk" or Group Names don't get cut off.
            # We push the bottom up to make room for the table (though bbox_inches='tight' helps, explicit layout is safer for alignment)
            plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)

            # Determine Ticks
            max_time = df_clean[time_col].max()
            if tick_interval:
                custom_ticks = np.arange(0, max_time + tick_interval, tick_interval)
                ax.set_xticks(custom_ticks)
                ax.set_xlim(0, custom_ticks[-1]) # Strict x-limits for table alignment
            else:
                 ax.set_xlim(left=0) # At least start at 0
             
            # Set Y-Axis Customization
            if y_tick_interval:
                 ax.set_yticks(np.arange(y_min, y_max + y_tick_interval/10, y_tick_interval))
            ax.set_ylim(y_min, y_max)

            if group_col != "None":
                # Sorting groups for consistent color assignment
                groups = sorted(df_clean[group_col].unique())
                results = []
            
                # Determine Color Palette
                palette = None
                if selected_theme in all_themes:
                    palette = all_themes[selected_theme]
            
                fitters = []
                plot_colors = []
                plot_labels = [] # For the risk table
            
                for i, group in enumerate(groups):
                    mask = df_clean[group_col] == group
                
                    # Resolve Color
                    color = None
                    if selected_theme == "Custom":
                        color = custom_colors.get(group)
                    elif palette:
                         color = palette[i % len(palette)]
                
                    if color is None:
                        # Fallback default cycle if no theme
                        color = f"C{i}"
                
                    plot_colors.append(color)
                
                    # Resolve Label
                    label = group_labels.get(group, str(group))
                    plot_labels.append(label)

                    kmf_group = KaplanMeierFitter()
                    kmf_group.fit(df_clean[time_col][mask], df_clean[event_col][mask], label=label)
                    kmf_group.plot_survival_function(ax=ax, ci_show=show_ci, show_censors=show_censored, color=color, linewidth=line_width)
                    fitters.append(kmf_group)
            
                # P-value and Legend if applicable (Single group usually no legend needed unless CI)
                if show_p_val_plot and p_value_text:
                     bbox_props = dict(facecolor='white', alpha=0.5, boxstyle='round') if show_p_val_box_main else None
                     ax.text(pval_x_main, pval_y_main, p_value_text, transform=ax.transAxes, ha='right', va='bottom', bbox=bbox_props, fontsize=p_val_fontsize)

                # Risk Table logic (basic implementation using lifelines built-in if possible, or custom)
                if show_risk_table:
                    # Custom add_at_risk_counts integration
                    # We need fitters for all to use add_at_risk_counts
                    # fitters list already populated above
                    add_at_risk_counts(fitters, ax=ax, y_shift=table_height, colors=plot_colors, labels=plot_labels)
                
                # Apply Custom Label
                ax.set_title(main_title, fontsize=title_fontsize, weight=title_fontweight)
                ax.set_xlabel(x_label, fontsize=axes_fontsize)
                if y_label:
                    ax.set_ylabel(y_label, fontsize=axes_fontsize)
                ax.tick_params(axis='both', which='major', labelsize=axes_fontsize)
                
                # Legend Customization
                if show_legend_main:
                     ax.legend(fontsize=legend_fontsize, loc=(leg_x_main, leg_y_main), frameon=show_legend_box_main)
                else:
                     if ax.get_legend():
                         ax.get_legend().remove()
                
                st.pyplot(fig)
                
                # Save to session_state for Report
                st.session_state['report_fig_km'] = fig

                # DOWNLOAD BUTTON
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
                buf.seek(0)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 Download Plot (300 DPI)",
                        data=buf,
                        file_name="survival_plot_300dpi.png",
                        mime="image/png"
                    )
                with col2:
                    buf_hi = io.BytesIO()
                    fig.savefig(buf_hi, format="png", dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
                    buf_hi.seek(0)
                    st.download_button(
                        label="💾 Download High-Res Plot (600 DPI)",
                        data=buf_hi,
                        file_name="survival_plot_600dpi.png",
                        mime="image/png"
                    )

                # 2. Statistics (Cox PH / Logrank)
                st.divider()
                st.subheader("Statistical Analysis")
            
                # Logrank Test
                if len(groups) >= 2:
                    result = multivariate_logrank_test(df_clean[time_col], df_clean[group_col], df_clean[event_col])
                    st.write(f"**Log-Rank Test p-value**: {result.p_value:.4f}")
            
                # Cox PH (Hazard Ratio)
                st.subheader("Cox Proportional Hazards (Hazard Ratios)")
            
                # Reference Group Selection
                unique_values = sorted(df_clean[group_col].unique())
                reference_group = st.selectbox("Select Reference Group (Baseline)", unique_values, index=0)
            
                # Prepare data for CoxPH
                cox_df = df_clean[[time_col, event_col, group_col]].dropna()
            
                try:
                    # 1. Prepare Data
                    # We need separate binary columns for each group except the reference
                    # This explicitly sets up "Treatment Coding" (Group X vs Reference)
                
                    # Get all unique groups
                    all_groups = sorted(df_clean[group_col].unique())
                
                    # Create a DataFrame for regression
                    cox_data = df_clean[[time_col, event_col]].copy()
                
                    # Create dummy variables manually to be 100% sure of the mapping
                    for group in all_groups:
                        if str(group) == str(reference_group):
                            continue
                        
                        # Create column: "Group_Value"
                        col_name = str(group)
                        # 1 if row belongs to this group, 0 otherwise
                        cox_data[col_name] = (df_clean[group_col] == group).astype(int)

                    # 2. Fit Model
                    # Standard CoxPH without penalizer to get unbiased estimates
                    cph = CoxPHFitter() 
                    cph.fit(cox_data, duration_col=time_col, event_col=event_col)
                
                    # 3. Display Results
                    summary_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
                    summary_df = summary_df.rename(columns={
                        'exp(coef)': 'Hazard Ratio (HR)', 
                        'exp(coef) lower 95%': 'Lower 95% CI', 
                        'exp(coef) upper 95%': 'Upper 95% CI',
                        'p': 'p-value'
                    })
                
                    st.dataframe(summary_df.style.format("{:.3f}"))
                    
                    # Download Cox Table
                    csv_cox = summary_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="💾 Download Cox HR Table",
                        data=csv_cox,
                        file_name="cox_ph_table.csv",
                        mime="text/csv"
                    )
                    st.caption(f"Reference Group: **{reference_group}** (All HRs are relative to this group)")

                except Exception as e:
                    st.error(f"Cox Model Error: {e}")
                    st.info("Tip: This often happens if a group has too few events or perfectly predicts survival (separation).")
                
                except Exception as e:
                    st.warning(f"Could not run Cox PH model: {e}")
                
                # --- MEDIAN SURVIVAL TABLE ---
                st.subheader("Median Survival Time")
                from lifelines.utils import median_survival_times
                
                median_data = []
                for group in groups:
                    mask = df_clean[group_col] == group
                    kmf_med = KaplanMeierFitter()
                    kmf_med.fit(df_clean[time_col][mask], df_clean[event_col][mask], label=str(group))
                    
                    median_os = kmf_med.median_survival_time_
                    
                    # 95% CI for Median
                    try:
                        median_ci_df = median_survival_times(kmf_med.confidence_interval_)
                        # The dataframe columns are usually named {label}_lower_0.95, {label}_upper_0.95
                        # But simpler is to index 0 if it's single row
                        lower = median_ci_df.iloc[0, 0]
                        upper = median_ci_df.iloc[0, 1]
                        ci_str = f"({lower:.1f} - {upper:.1f})"
                    except:
                        ci_str = "(NR - NR)"
                    
                    # Formatting text
                    med_str = f"{median_os:.1f}" if not np.isinf(median_os) else "NR" # NR = Not Reached
                    
                    label = group_labels.get(group, str(group))
                    median_data.append({
                        "Group": label,
                        "Median Survival": med_str,
                        "95% CI (Median)": ci_str
                    })
                
                median_df = pd.DataFrame(median_data)
                st.table(median_df.style.format())
                
                # Download Median Table
                csv_med = median_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Median Survival Table",
                    data=csv_med,
                    file_name="median_survival_table.csv",
                    mime="text/csv"
                )

                # Point-in-Time Survival Estimates
                st.subheader("Point-in-Time Survival Estimates")
                st.write("Calculate survival probability at a specific time (e.g., 2-year OS).")
            
                target_time = st.number_input("Enter Time Point (e.g., 24 months)", min_value=0.0, value=24.0, step=6.0)
            
                est_data = []
                for group in groups:
                     mask = df_clean[group_col] == group
                     kmf_est = KaplanMeierFitter()
                     kmf_est.fit(df_clean[time_col][mask], df_clean[event_col][mask], label=str(group))
                 
                     # Survival at t
                     surv_prob = kmf_est.survival_function_at_times(target_time).iloc[0]
                 
                     # CI at t (Interpolate from the confidence interval dataframe)
                     # kmf_est.confidence_interval_survival_function_ is a DataFrame with indices as timeline
                     ci_df = kmf_est.confidence_interval_survival_function_
                     # We need to find the correct values for target_time. 
                     # Since it's a step function, we take the value at the largest time index <= target_time
                     # Reindexing or forward filling is a good way.
                 
                     # Append target_time to index, sort, ffill, then loc
                     combined_index = ci_df.index.union([target_time]).sort_values()
                     ci_df_interp = ci_df.reindex(combined_index).ffill()
                 
                     try:
                        lower = ci_df_interp.loc[target_time].iloc[0]
                        upper = ci_df_interp.loc[target_time].iloc[1]
                     except:
                        lower = 0
                        upper = 0
                 
                     label = group_labels.get(group, str(group))
                     est_data.append({
                         "Group": label,
                         f"Survival at {target_time}": f"{surv_prob:.1%}",
                         "95% CI": f"({lower:.1%} - {upper:.1%})"
                     })
            
                est_df = pd.DataFrame(est_data)
                st.table(est_df)
                
                # Download Point-in-Time Table
                csv_pit = est_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"💾 Download Survival Estimates at {target_time}",
                    data=csv_pit,
                    file_name=f"survival_at_{target_time}.csv",
                    mime="text/csv"
                )
                
                # --- PAIRWISE COMPARISONS ---
                st.subheader("Pairwise Log-Rank Comparisons")
                from lifelines.statistics import pairwise_logrank_test
                
                if len(groups) > 2:
                    st.write("Comparison between specific pairs of groups (p-values).")
                    
                    # Run Pairwise Test
                    results = pairwise_logrank_test(df_clean[time_col], df_clean[group_col], df_clean[event_col])
                    
                    # The results summary is a rich dataframe, but we want a matrix or list
                    # summary_df contains p-values
                    pairwise_df = results.summary
                    
                    # Formatting for display
                    st.dataframe(pairwise_df.style.format({"p": "{:.4f}"}))
                    
                    # Download Pairwise Table
                    csv_pair = pairwise_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="💾 Download Pairwise Comparison Table",
                        data=csv_pair,
                        file_name="pairwise_logrank.csv",
                        mime="text/csv"
                    )
                elif len(groups) == 2:
                    st.info("Pairwise comparison is identical to the Global Log-Rank test for 2 groups.")

                # --- AI NARRATOR (Univariable) ---
                st.divider()
                st.subheader("🤖 AI Result Narrator")
                if st.button("Generate Summary Text (Univariable)"):
                    # 1. Significance
                    sig_word = "significantly" if result.p_value < 0.05 else "not significantly"
                    
                    # 2. Main statement
                    summary = f"The survival analysis comparing groups defined by **{group_col}** ({', '.join([str(g) for g in groups])}) revealed that {group_col} was **{sig_word} associated with survival** (Log-rank p={result.p_value:.4f}). "
                    
                    # 3. Median details
                    med_details = []
                    for dataItem in median_data:
                        med_details.append(f"{dataItem['Group']} (Median: {dataItem['Median Survival']}, 95% CI: {dataItem['95% CI (Median)']})")
                    
                    summary += "Median survival times were: " + "; ".join(med_details) + "."
                    
                    st.success("Summary Generated:")
                    st.code(summary, language="text")

            else:
                # Single group
                color = None
                if selected_theme in all_themes and len(all_themes[selected_theme]) > 0:
                     color = all_themes[selected_theme][0]
                elif selected_theme == "Custom":
                     color = st.sidebar.color_picker("Color for All Patients", "#1f77b4")

                kmf_all = KaplanMeierFitter()
                kmf_all.fit(df_clean[time_col], df_clean[event_col], label="All Patients")
                kmf_all.plot_survival_function(ax=ax, ci_show=show_ci, show_censors=show_censored, color=color, linewidth=line_width)
                
                if show_risk_table:
                    # Define lists for single group to match function variable names
                    plot_colors = [color] if color else None
                    plot_labels = ["All Patients"]
                    # from lifelines.plotting import add_at_risk_counts (REMOVED due to bug)
                    add_at_risk_counts([kmf_all], ax=ax, y_shift=table_height, colors=plot_colors, labels=plot_labels)
            
                # Apply Custom Label
                ax.set_title(main_title, fontsize=title_fontsize, weight=title_fontweight)
                ax.set_xlabel(x_label, fontsize=axes_fontsize)
                if y_label:
                    ax.set_ylabel(y_label, fontsize=axes_fontsize)
                ax.tick_params(axis='both', which='major', labelsize=axes_fontsize)
                
                if show_legend_main:
                     ax.legend(fontsize=legend_fontsize, loc=(leg_x_main, leg_y_main), frameon=show_legend_box_main)
                else:
                     if ax.get_legend():
                         ax.get_legend().remove()
                
                if show_p_val_plot and p_value_text:
                     bbox_props = dict(facecolor='white', alpha=0.5, boxstyle='round') if show_p_val_box_main else None
                     ax.text(pval_x_main, pval_y_main, p_value_text, transform=ax.transAxes, ha='right', va='bottom', bbox=bbox_props, fontsize=p_val_fontsize)
                
                if main_free_text:
                     bbox_props = dict(facecolor='white', alpha=0.5, boxstyle='round') if main_text_box else None
                     ax.text(main_text_x, main_text_y, main_free_text, transform=ax.transAxes, ha='center', va='center', bbox=bbox_props, fontsize=main_text_size)
                
                st.pyplot(fig)

                # DOWNLOAD BUTTON
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
                buf.seek(0)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 Download Plot (300 DPI)",
                        data=buf,
                        file_name="survival_plot_300dpi.png",
                        mime="image/png"
                    )
                with col2:
                    buf_hi = io.BytesIO()
                    fig.savefig(buf_hi, format="png", dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
                    buf_hi.seek(0)
                    st.download_button(
                        label="💾 Download High-Res Plot (600 DPI)",
                        data=buf_hi,
                        file_name="survival_plot_600dpi.png",
                        mime="image/png"
                    )


        with tab2:
            st.subheader("Multivariable Cox Proportional Hazards")
            st.write("Perform a multivariable analysis to assess independent prognostic factors.")
            
            # 1. Select Covariates
            # Exclude Time and Event columns from options
            covariate_options = [c for c in columns if c not in [time_col, event_col]]
            covariates = st.multiselect("Select Covariates for Analysis", covariate_options)
            
            if covariates:
                st.info(f"Target: **{time_col}** (Time), **{event_col}** (Event)")
                
                # Check for NaNs in selected variables
                mv_cols = [time_col, event_col] + covariates
                mv_df = df[mv_cols].dropna()
                dropped_mv = len(df) - len(mv_df)
                
                if dropped_mv > 0:
                    st.warning(f"⚠️ {dropped_mv} rows will be dropped due to missing values in the selected covariates. Analysis based on {len(mv_df)} rows.")
                
                if len(mv_df) < 10:
                     st.error("Not enough data points for multivariable analysis.")
                else:
                    # --- Reference Group Selection for Categorical Variables ---
                    categorical_refs = {}
                    
                    # 1. Identify Categorical Columns
                    cat_cols = [c for c in covariates if pd.api.types.is_object_dtype(mv_df[c]) or pd.api.types.is_categorical_dtype(mv_df[c])]
                    
                    if cat_cols:
                        st.markdown("##### Reference Group Selection")
                        cols = st.columns(min(3, len(cat_cols)))
                        for i, col in enumerate(cat_cols):
                            unique_levels = sorted(mv_df[col].dropna().unique().astype(str))
                            with cols[i % 3]:
                                # User selects one level to be the reference (dropped)
                                ref = st.selectbox(f"Ref for {col}", unique_levels, key=f"ref_{col}", index=0)
                                categorical_refs[col] = ref

                    run_analysis = st.button("Run Multivariable Analysis")
                    
                    if run_analysis:
                        st.session_state['mv_analysis_active'] = True
                        
                    if st.session_state.get('mv_analysis_active', False):
                        try:
                            # Encore Categorical Variables MANUALLY to handle Reference Group
                            mv_data_encoded = mv_df.copy()
                            
                            # Drop original categorical columns from encoding base, we will add dummies
                            mv_data_encoded = mv_data_encoded.drop(columns=cat_cols)
                            
                            for col in cat_cols:
                                ref = categorical_refs.get(col)
                                # Get Dummies
                                dummies = pd.get_dummies(mv_df[col], prefix=col)
                                
                                # Drop the reference column
                                ref_col_name = f"{col}_{ref}"
                                if ref_col_name in dummies.columns:
                                    dummies = dummies.drop(columns=[ref_col_name])
                                    
                                # Concatenate
                                mv_data_encoded = pd.concat([mv_data_encoded, dummies], axis=1)
                            
                            # Sanitize Column Names for Lifelines/Stats (remove spaces/special chars)
                            # This is important for formula strings but CoxPHFitter handles dataframe input well.
                            # However, cleaner names are better for the plot.
                            mv_data_encoded.columns = [c.replace(' ', '_').replace('+', 'pos').replace('-', 'neg') for c in mv_data_encoded.columns]
                            
                            # Fit Model
                            cph_mv = CoxPHFitter()
                            cph_mv.fit(mv_data_encoded, duration_col=time_col, event_col=event_col)
                            
                            # Results Table
                            summary_mv = cph_mv.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
                            summary_mv.columns = ['Hazard Ratio (HR)', 'Lower 95%', 'Upper 95%', 'p-value']
                            
                            st.write("### Regression Results")
                            st.dataframe(summary_mv.style.format("{:.3f}"))
                            
                            # Save to Session State specifically for Narrator or Persistence
                            st.session_state['mv_summary_df'] = summary_mv

                            # Forest Plot
                            st.write("### Forest Plot")
                            
                            # Prepare Data for Plot
                            plot_data = summary_mv.copy()
                            plot_data = plot_data.sort_index(ascending=False) # Top to bottom on plot
                            
                            # Dynamic height
                            fig_forest, ax_forest = plt.subplots(figsize=(10, max(4, len(plot_data) * 0.5 + 1)))
                            
                            # Theme Color
                            forest_color = '#1f77b4'
                            if selected_theme in all_themes and len(all_themes[selected_theme]) > 0:
                                 forest_color = all_themes[selected_theme][0]
                            elif selected_theme == "Custom":
                                 pass # Use default blue or allow custom override? Basic blue is fine.
                            
                            y_pos = np.arange(len(plot_data))
                            
                            # Plot Points and Error Bars
                            x_errs = [
                                plot_data['Hazard Ratio (HR)'] - plot_data['Lower 95%'],  # Updated col names
                                plot_data['Upper 95%'] - plot_data['Hazard Ratio (HR)']
                            ]
                            
                            ax_forest.errorbar(plot_data['Hazard Ratio (HR)'], y_pos, xerr=x_errs, 
                                               fmt='o', color=forest_color, ecolor='black', capsize=5, markersize=8)
                            
                            # Reference Line
                            ax_forest.axvline(x=1, color='red', linestyle='--', linewidth=1)
                            
                            # Labels
                            ax_forest.set_yticks(y_pos)
                            ax_forest.set_yticklabels(plot_data.index, fontsize=10, fontweight='bold')
                            ax_forest.set_xlabel("Hazard Ratio (95% CI)")
                            ax_forest.set_title("Multivariable Cox Regression Results")
                            
                            # Grid
                            ax_forest.grid(True, axis='x', linestyle=':', alpha=0.6)
                            
                            # Clean Spines
                            ax_forest.spines['top'].set_visible(False)
                            ax_forest.spines['right'].set_visible(False)
                            ax_forest.spines['left'].set_visible(False)
                            
                            st.pyplot(fig_forest)
                            
                            # Save to session_state for Report
                            st.session_state['report_fig_forest'] = fig_forest
                            
                            # Download
                            buf_forest = io.BytesIO()
                            fig_forest.savefig(buf_forest, format="png", dpi=300, bbox_inches='tight')
                            buf_forest.seek(0)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button("💾 Download Forest Plot (300 DPI)", buf_forest, "forest_plot_300dpi.png", "image/png")
                            with col2:
                                buf_forest_hi = io.BytesIO()
                                fig_forest.savefig(buf_forest_hi, format="png", dpi=600, bbox_inches='tight')
                                buf_forest_hi.seek(0)
                                st.download_button("💾 Download High-Res Forest Plot (600 DPI)", buf_forest_hi, "forest_plot_600dpi.png", "image/png")
                                
                                # --- AI NARRATOR (Multivariable) ---
                                st.divider()
                                st.write("### 🤖 AI Result Narrator")
                                
                                # Use session state DF if available (handles button consistency if needed)
                                mv_df_for_narrator = st.session_state.get('mv_summary_df', summary_mv)
                                
                                if st.button("Generate Summary Text (Multivariable)"):
                                    mv_summary = "In the multivariable Cox regression model adjusted for relevant covariates, the following associations were observed:\n\n"
                                    
                                    # Iterate over rows
                                    for idx, row in mv_df_for_narrator.iterrows():
                                        hr = row['Hazard Ratio (HR)']
                                        p = row['p-value']
                                        ci_low = row['Lower 95%'] # Updated key
                                        ci_high = row['Upper 95%'] # Updated key
                                        
                                        sig_txt = "significantly associated" if p < 0.05 else "not significantly associated"
                                        
                                        mv_summary += f"* **{idx}**: {sig_txt} with the event (HR={hr:.2f}, 95% CI {ci_low:.2f}-{ci_high:.2f}, p={p:.4f}).\n"
                                        
                                    st.success("Summary Generated:")
                                    st.text_area("Copy this text:", value=mv_summary, height=200)
                            
                        except Exception as e:
                            st.error(f"Error running model: {e}")
                            st.info("Ensure you are not including variables that perfectly predict the outcome (separation).")
                            



            else:
                st.info("Select at least one covariate variable (e.g., Age, Gender, Mutations) to begin.")

        with tab3:
            st.subheader("Competing Risks Analysis (Cumulative Incidence)")
            st.write("Calculate Cumulative Incidence Function (CIF) considering competing events (e.g., Relapse vs Death).")
            
            # Logic Selection: Single Composite Column vs Two Separate Columns
            cif_mode = st.radio("Data Format:", ["Single 'Status' Column (with multiple codes)", "Two Separate Columns (e.g., Relapse & OS)"], index=1)
            
            cif_df = None
            cif_time_col = None
            cif_event_col = None
            cif_event_of_interest = None
            
            if cif_mode == "Single 'Status' Column (with multiple codes)":
                st.info(f"Using Target Time Column: **{time_col}**")
                st.info(f"Using Target Event Column: **{event_col}**")
                
                # Identify Event Codes
                unique_events = sorted(df_clean[event_col].unique())
                st.write(f"**Observed Event Codes:** {unique_events}")
                
                col1, col2 = st.columns(2)
                with col1:
                    event_of_interest = st.selectbox("Select Event of Interest", unique_events, index=1 if len(unique_events) > 1 else 0)
                
                cif_df = df_clean.copy()
                cif_time_col = time_col
                cif_event_col = event_col
                cif_event_of_interest = event_of_interest
                
            else:
                # Two Column Logic
                st.write("Construct a Competing Risk endpoint from two separate events (e.g., Relapse vs Death).")
                st.markdown("""
                **Logic:**
                *   **Event 1 (Interest)**: Only counts if it happens *first*.
                *   **Event 2 (Competing)**: Counts if it happens *first* (and prevents Event 1).
                *   **Censored**: If neither happens.
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Event 1 (e.g., Relapse)")
                    rfs_time_col = st.selectbox("Time Column", columns, index=columns.index("RFS_Days") if "RFS_Days" in columns else 0, key="rfs_time")
                    rfs_stat_col = st.selectbox("Status Column (1=Event)", columns, index=columns.index("RFS_Status") if "RFS_Status" in columns else 0, key="rfs_stat")
                    
                with col2:
                    st.markdown("#### Event 2 (Competing, e.g., OS)")
                    os_time_col = st.selectbox("Time Column", columns, index=columns.index("OS_Days") if "OS_Days" in columns else 0, key="os_time")
                    os_stat_col = st.selectbox("Status Column (1=Event)", columns, index=columns.index("OS_Status") if "OS_Status" in columns else 0, key="os_stat")

                if st.button("Construct & Analyze"):
                    # Data Wrangling
                    temp_df = df[[rfs_time_col, rfs_stat_col, os_time_col, os_stat_col]].copy()
                    if group_col != "None":
                        temp_df[group_col] = df[group_col]
                    
                    temp_df = temp_df.dropna()
                    
                    # Create Composite Columns
                    temp_df['Composite_Time'] = temp_df[[rfs_time_col, os_time_col]].min(axis=1)
                    
                    # Status Logic:
                    # 0 = Censored (Both 0)
                    # 1 = Event 1 (RFS happened AND (OS didn't happen OR RFS happened before/at same time as OS))
                    # 2 = Event 2 (OS happened AND (RFS didn't happen OR OS happened before RFS))
                    
                    def get_status(row):
                        t_rfs = row[rfs_time_col]
                        s_rfs = row[rfs_stat_col]
                        t_os = row[os_time_col]
                        s_os = row[os_stat_col]
                        
                        # If both censored
                        if s_rfs == 0 and s_os == 0:
                            return 0 # Censored
                        
                        # If only RFS event
                        if s_rfs == 1 and s_os == 0:
                            return 1 # Relapse
                            
                        # If only OS event
                        if s_rfs == 0 and s_os == 1:
                            return 2 # Death
                            
                        # If BOTH events (Competing risk scenario logic)
                        if s_rfs == 1 and s_os == 1:
                            if t_rfs <= t_os:
                                return 1 # Relapse first
                            else:
                                return 2 # Death first (unlikely for RFS but possible for other events)
                        
                        return 0

                    temp_df['Composite_Status'] = temp_df.apply(get_status, axis=1)
                    
                    # Persist in Session State
                    st.session_state['two_col_cif_df'] = temp_df
                    st.session_state['two_col_cif_time'] = 'Composite_Time'
                    st.session_state['two_col_cif_event'] = 'Composite_Status'
                    st.session_state['two_col_cif_interest'] = 1
                
                # Retrieve from Session State if available
                if 'two_col_cif_df' in st.session_state:
                    cif_df = st.session_state['two_col_cif_df']
                    cif_time_col = st.session_state['two_col_cif_time']
                    cif_event_col = st.session_state['two_col_cif_event']
                    cif_event_of_interest = st.session_state['two_col_cif_interest']
                    
                    st.success(f"Constructed composite endpoint. Found {sum(cif_df['Composite_Status']==1)} primary events and {sum(cif_df['Composite_Status']==2)} competing events.")

            # Plot CIF
            if cif_df is not None: # Triggered either by default mode or button in 2-col mode
                
                # Check groupings
                cif_fitters = []
                cif_colors = []
                cif_labels = []
                
                # Force Font Update
                plt.rcParams['font.family'] = selected_font
                fig_cif, ax_cif = plt.subplots(figsize=(plot_width, plot_height))
                
                # Layout adjustments for Risk Table
                plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
                
                # Determine Ticks (Uniform with Main Plot)
                cif_max_time = cif_df[cif_time_col].max()
                if tick_interval:
                    cif_custom_ticks = np.arange(0, cif_max_time + tick_interval, tick_interval)
                    ax_cif.set_xticks(cif_custom_ticks)
                    ax_cif.set_xlim(0, cif_custom_ticks[-1]) # Strict x-limits for table alignment
                else:
                     ax_cif.set_xlim(left=0) # At least start at 0
                
                # Fine-Gray Analysis (for P-value and Table)
                fg_p_value_text = ""
                fg_summary = None
                
                if group_col != "None" and group_col in cif_df.columns:
                     unique_grps = sorted(cif_df[group_col].dropna().unique())
                     if len(unique_grps) >= 2:
                          try:
                              # Select Reference Group (Baseline)
                              st.write("**Fine-Gray Reference Group (Baseline)**")
                              fg_ref_group = st.selectbox("Select Baseline Group for Subdistribution HR", unique_grps, index=0, key="fg_ref_group")
                              
                              with st.spinner("Calculating Fine-Gray Statistics..."):
                                  # 1. Prepare Weighted Data
                                  fg_data = compute_fine_gray_weights(cif_df, cif_time_col, cif_event_col, cif_event_of_interest)
                                  
                                  # 2. Encode Group Variable (One-Hot) with Custom Reference
                                  # To set a specific reference, we can use pd.get_dummies and drop the reference column
                                  fg_data_encoded = pd.get_dummies(fg_data, columns=[group_col], drop_first=False) # Keep all initially
                                  
                                  # Drop the reference group column
                                  ref_col_name = f"{group_col}_{fg_ref_group}"
                                  if ref_col_name in fg_data_encoded.columns:
                                      fg_data_encoded = fg_data_encoded.drop(columns=[ref_col_name])
                                  
                                  # Select columns: 'start', 'stop', event(status), weights, id, and the new dummy columns
                                  # The remaining dummy columns are the comparisons vs reference
                                  dummy_cols = [c for c in fg_data_encoded.columns if c.startswith(f"{group_col}_")]
                                  cols_to_fit = ['start', 'stop', 'status', 'weight', 'id'] + dummy_cols
                                 
                                  # 3. Fit Fine-Gray Model (Weighted Cox)
                                  # IMPORTANT: The weighted dataframe from compute_fine_gray_weights is in counting process format (start, stop).
                                  # We must NOT use duration_col=cif_time_col.
                                  cph_fg = CoxPHFitter()
                                  cph_fg.fit(fg_data_encoded[cols_to_fit], 
                                             duration_col='stop', entry_col='start', event_col='status', weights_col='weight', 
                                             cluster_col='id', robust=True)
                                  
                                  # 4. Extract P-value (Gray's Test Equivalent)
                                  # Log-Likelihood Ratio Test against null model
                                  res_fg = cph_fg.log_likelihood_ratio_test()
                                  # Log-Likelihood Ratio Test against null model
                                  res_fg = cph_fg.log_likelihood_ratio_test()
                                  if show_p_val_plot_cif:
                                      fg_p_value_text = f"Gray's p = {res_fg.p_value:.4f}"
                                  
                                  # 5. Extract HR Table
                                  fg_summary = cph_fg.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
                                  fg_summary.columns = ['Subdist HR', 'Lower 95%', 'Upper 95%', 'p-value']
                                 
                          except Exception as e:
                             st.error(f"Fine-Gray Analysis Failed: {e}")
                             # Fallback to simple Cause-Specific Log-Rank for simple plot label if FG fails
                             try:
                                  temp_evt = (cif_df[cif_event_col] == cif_event_of_interest).astype(int)
                                  res_cif = multivariate_logrank_test(cif_df[cif_time_col], cif_df[group_col], temp_evt)
                                  if show_p_val_plot_cif:
                                    fg_p_value_text = f"CS-LogRank p = {res_cif.p_value:.4f}"
                             except:
                                 pass
                
                # Display P-value on Plot
                if show_p_val_plot and fg_p_value_text:
                     bbox_props = dict(facecolor='white', alpha=0.5, boxstyle='round') if show_p_val_box_cif else None
                     ax_cif.text(pval_x_cif, pval_y_cif, fg_p_value_text, transform=ax_cif.transAxes, ha='right', va='bottom', bbox=bbox_props, fontsize=p_val_fontsize)

                if group_col != "None" and group_col in cif_df.columns:
                    groups = sorted(cif_df[group_col].unique())
                    for i, group in enumerate(groups):
                        mask = cif_df[group_col] == group
                        
                        # Resolve Color
                        color = None
                        if selected_theme == "Custom":
                            color = custom_colors.get(group)
                        elif selected_theme in all_themes:
                             palette = all_themes[selected_theme]
                             color = palette[i % len(palette)]
                        
                        if color is None:
                            color = f"C{i}"
                        
                        cif_colors.append(color)
                        
                        # Resolve Label
                        # Note: group_labels dictionary keys are from the ORIGINAL dataframe
                        # If Two-Column mode preserved the group values, this works.
                        label = group_labels.get(group, str(group))
                        cif_labels.append(label)
                            
                        # Fit Aalen-Johansen
                        ajf = AalenJohansenFitter(calculate_variance=True)
                        ajf.fit(cif_df[cif_time_col][mask], cif_df[cif_event_col][mask], event_of_interest=cif_event_of_interest, label=label)
                        ajf.plot(ax=ax_cif, ci_show=show_ci, show_censors=False, color=color, linewidth=line_width) # Disable built-in to avoid error
                        cif_fitters.append(ajf)
                        
                        # Manual Censoring Ticks
                        if show_censored:
                            # 1. Identify censored times for this group
                            # In our logic (Composite or Single), 0 usually means censored
                            censored_mask = (cif_df[cif_event_col][mask] == 0)
                            censored_times = cif_df[cif_time_col][mask][censored_mask].values
                            
                            if len(censored_times) > 0:
                                # 2. Get CIF values at these times
                                # Reindex the CIF dict to find the value at or before the censored time (step function)
                                cif_line = ajf.cumulative_density_
                                # We need to handle step function behavior: forward fill
                                # Add censored times to the index, sort, ffill
                                combined_index = cif_line.index.union(censored_times).unique().sort_values()
                                cif_interp = cif_line.reindex(combined_index).ffill()
                                
                                # Extract y-values
                                # cif_interp is a DataFrame with one column (usually label name or unique ID)
                                y_values = cif_interp.loc[censored_times].iloc[:, 0].values
                                
                                # 3. Plot ticks
                                ax_cif.plot(censored_times, y_values, '|', color=color, markersize=10, markeredgewidth=1)

                else:
                    # Single Group
                     color = None
                     if selected_theme in all_themes and len(all_themes[selected_theme]) > 0:
                         color = all_themes[selected_theme][0]
                     elif selected_theme == "Custom":
                         color = st.sidebar.color_picker("Color for All Patients (CIF)", "#1f77b4")
                     
                     ajf = AalenJohansenFitter(calculate_variance=True)
                     ajf.fit(cif_df[cif_time_col], cif_df[cif_event_col], event_of_interest=cif_event_of_interest, label="All Patients")
                     ajf.plot(ax=ax_cif, ci_show=show_ci, show_censors=False, color=color, linewidth=line_width) # Disable built-in
                     
                     cif_fitters.append(ajf)
                     cif_colors.append(color)
                     cif_labels.append("All Patients")
                     
                     # Manual Censoring Ticks (Single Group)
                     if show_censored:
                        censored_mask = (cif_df[cif_event_col] == 0)
                        censored_times = cif_df[cif_time_col][censored_mask].values
                        if len(censored_times) > 0:
                            cif_line = ajf.cumulative_density_
                            combined_index = cif_line.index.union(censored_times).unique().sort_values()
                            cif_interp = cif_line.reindex(combined_index).ffill()
                            y_values = cif_interp.loc[censored_times].iloc[:, 0].values
                            ax_cif.plot(censored_times, y_values, '|', color=color, markersize=10, markeredgewidth=1)
                     
                # Apply Custom Label (CIF)
                ax_cif.set_title(cif_title, fontsize=title_fontsize, weight=title_fontweight)
                ax_cif.set_xlabel(x_label, fontsize=axes_fontsize)
                ax_cif.set_ylabel(cif_y_label, fontsize=axes_fontsize)
                
                # CIF Y-Axis Customization
                if cif_y_tick_interval:
                     ax_cif.set_yticks(np.arange(cif_y_min, cif_y_max + cif_y_tick_interval/10, cif_y_tick_interval))
                ax_cif.set_ylim(cif_y_min, cif_y_max)
                
                # Style adjustments
                fig_cif.patch.set_facecolor(plot_bgcolor)
                ax_cif.set_facecolor(plot_bgcolor)
                
                # Add Risk Table (Point-in-Time)
                if show_legend_cif:
                     ax_cif.legend(fontsize=legend_fontsize, loc=(leg_x_cif, leg_y_cif), frameon=show_legend_box_cif)
                else:
                     if ax_cif.get_legend():
                         ax_cif.get_legend().remove()
                
                if show_p_val_plot and fg_p_value_text:
                     bbox_props = dict(facecolor='white', alpha=0.5, boxstyle='round') if show_p_val_box_cif else None
                     ax_cif.text(pval_x_cif, pval_y_cif, fg_p_value_text, transform=ax_cif.transAxes, ha='right', va='bottom', bbox=bbox_props, fontsize=p_val_fontsize)
                # Add Risk Table
                if show_risk_table:
                    add_at_risk_counts(cif_fitters, ax=ax_cif, y_shift=table_height, colors=cif_colors, labels=cif_labels)
                
                ax_cif.set_xlabel(x_label, fontsize=axes_fontsize)
                if cif_y_label:
                    ax_cif.set_ylabel(cif_y_label, fontsize=axes_fontsize)
                ax_cif.tick_params(axis='both', which='major', labelsize=axes_fontsize)
                
                if show_legend_cif:
                     ax_cif.legend(fontsize=legend_fontsize, loc=(leg_x_cif, leg_y_cif), frameon=show_legend_box_cif)
                else:
                     if ax_cif.get_legend():
                         ax_cif.get_legend().remove()

                if cif_free_text:
                     bbox_props = dict(facecolor='white', alpha=0.5, boxstyle='round') if cif_text_box else None
                     ax_cif.text(cif_text_x, cif_text_y, cif_free_text, transform=ax_cif.transAxes, ha='center', va='center', bbox=bbox_props, fontsize=cif_text_size)



                st.pyplot(fig_cif)
                
                # Download
                buf_cif = io.BytesIO()
                fig_cif.savefig(buf_cif, format="png", dpi=300, bbox_inches='tight', facecolor=fig_cif.get_facecolor(), edgecolor='none')
                buf_cif.seek(0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("💾 Download CIF Plot (300 DPI)", buf_cif, "cif_plot_300dpi.png", "image/png")
                with col2:
                    buf_cif_hi = io.BytesIO()
                    fig_cif.savefig(buf_cif_hi, format="png", dpi=600, bbox_inches='tight', facecolor=fig_cif.get_facecolor(), edgecolor='none')
                    buf_cif_hi.seek(0)
                    st.download_button("💾 Download High-Res CIF Plot (600 DPI)", buf_cif_hi, "cif_plot_600dpi.png", "image/png")
                
                # Display Fine-Gray Table
                if fg_summary is not None:
                    st.write("### Subdistribution Hazard Ratios (Fine-Gray)")
                    st.write("Covariate effect on the cumulative incidence of the event of interest, accounting for competing risks.")
                    st.dataframe(fg_summary.style.format("{:.3f}"))
                
                # Point-in-Time Cumulative Incidence Estimates
                st.subheader("Point-in-Time Cumulative Incidence")
                st.write("Calculate cumulative incidence probability at a specific time.")
            
                cif_target_time = st.number_input("Enter Time Point (e.g., 24 months)", min_value=0.0, value=24.0, step=6.0, key="cif_time_input")
            
                cif_est_data = []
                for ajf, label in zip(cif_fitters, cif_labels):
                     
                     # Cumulative Incidence at t
                     # ajf.cumulative_density_ is a DataFrame with index as timeline
                     cif_line = ajf.cumulative_density_
                     
                     # CI at t
                     ci_df = ajf.confidence_interval_
                     
                     # Interpolation Logic
                     # Append target_time to index, sort, ffill, then loc
                     combined_index = cif_line.index.union([cif_target_time]).sort_values()
                     
                     cif_line_interp = cif_line.reindex(combined_index).ffill()
                     ci_df_interp = ci_df.reindex(combined_index).ffill()
                     
                     try:
                        # Extract CIP
                        cip_val = cif_line_interp.loc[cif_target_time].iloc[0]
                        
                        # Extract CI (Lower, Upper)
                        lower = ci_df_interp.loc[cif_target_time].iloc[0]
                        upper = ci_df_interp.loc[cif_target_time].iloc[1]
                     except:
                        cip_val = 0
                        lower = 0
                        upper = 0
                 
                     cif_est_data.append({
                         "Group": label,
                         f"Cumulative Incidence at {cif_target_time}": f"{cip_val:.1%}",
                         "95% CI": f"({lower:.1%} - {upper:.1%})"
                     })
            
                cif_est_df = pd.DataFrame(cif_est_data)
                st.table(cif_est_df)
                
                # Download Point-in-Time Table
                csv_cif_pit = cif_est_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"💾 Download Inc. Estimates at {cif_target_time}",
                    data=csv_cif_pit,
                    file_name=f"cif_estimates_at_{cif_target_time}.csv",
                    mime="text/csv"
                )
                
                # --- MEDIAN TIME TO INCIDENCE ---
                st.subheader("Median Time to Incidence (50% Event Prob.)")
                
                cif_median_data = []
                for ajf, label in zip(cif_fitters, cif_labels):
                    cif_line = ajf.cumulative_density_
                    series = cif_line.iloc[:, 0]
                    
                    if series.max() >= 0.5:
                        median_time = series[series >= 0.5].index[0]
                        med_str = f"{median_time:.1f}"
                    else:
                        med_str = "NR" 
                    
                    # 95% CI for Median Time
                    # Logic: Time range where 95% CI of Probability contains 0.5
                    # Lower Time Bound comes from Upper Probability Bound crossing 0.5
                    # Upper Time Bound comes from Lower Probability Bound crossing 0.5
                    try:
                         ci_df = ajf.confidence_interval_
                         # Columns: [label]_lower_0.95, [label]_upper_0.95 (order may vary, but lower is first)
                         # Usually in lifelines, CI columns are {label}_lower_0.95 and {label}_upper_0.95
                         # We can select by position: 0=Lower Prob, 1=Upper Prob
                         
                         prob_lower = ci_df.iloc[:, 0]
                         prob_upper = ci_df.iloc[:, 1]
                         
                         # Time Lower Limit = Time when Prob UPPER bound >= 0.5
                         if prob_upper.max() >= 0.5:
                              time_lower = prob_upper[prob_upper >= 0.5].index[0]
                              time_lower_str = f"{time_lower:.1f}"
                         else:
                              time_lower_str = "NR"
                              
                         # Time Upper Limit = Time when Prob LOWER bound >= 0.5
                         if prob_lower.max() >= 0.5:
                              time_upper = prob_lower[prob_lower >= 0.5].index[0]
                              time_upper_str = f"{time_upper:.1f}"
                         else:
                              time_upper_str = "NR"
                         
                         if time_lower_str == "NR" and time_upper_str == "NR":
                              ci_str = "(NR - NR)"
                         else:
                              ci_str = f"({time_lower_str} - {time_upper_str})"
                    except:
                         ci_str = "(NR - NR)"

                    cif_median_data.append({
                        "Group": label,
                        "Median Time to Incidence": med_str,
                        "95% CI (Median)": ci_str
                    })
                
                cif_median_df = pd.DataFrame(cif_median_data)
                st.table(cif_median_df)
                
                csv_cif_med = cif_median_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Median Incidence Table",
                    data=csv_cif_med,
                    file_name="median_incidence_table.csv",
                    mime="text/csv"
                )

                # --- AI NARRATOR (CIF) ---
                st.divider()
                st.write("### 🤖 AI Result Narrator (Competing Risks)")
                if st.button("Generate Summary Text (CIF)"):
                     
                     cif_summary = "In the competing risks analysis using the Aalen-Johansen estimator:\n\n"
                     
                     # 1. Median Time Stats
                     if cif_median_data:
                         med_phrases = []
                         for item in cif_median_data:
                             med_phrases.append(f"{item['Group']} had a median time to incidence of {item['Median Time to Incidence']} (95% CI: {item['95% CI (Median)']})")
                         cif_summary += "* **Median Time to Incidence**: " + "; ".join(med_phrases) + ".\n"
                     
                     # 2. Point in Time Stats
                     if cif_est_data:
                         cif_summary += f"* **Cumulative Incidence at {cif_target_time}**: "
                         pit_phrases = []
                         col_name_pit = f"Cumulative Incidence at {cif_target_time}"
                         for item in cif_est_data:
                             val = item.get(col_name_pit, "N/A")
                             ci = item.get("95% CI", "")
                             pit_phrases.append(f"{item['Group']} {val} (95% CI {ci})")
                         cif_summary += "; ".join(pit_phrases) + ".\n"
                     
                     # 3. Fine-Gray Results
                     if fg_summary is not None:
                         cif_summary += "\n**Fine-Gray Regression Results** (accounting for competing risks):\n"
                         for idx, row in fg_summary.iterrows():
                             # Columns were renamed to ['Subdist HR', 'Lower 95%', 'Upper 95%', 'p-value']
                             hr = row['Subdist HR']
                             p = row['p-value']
                             low = row['Lower 95%']
                             high = row['Upper 95%']
                             sig_txt = "significantly associated" if p < 0.05 else "not significantly associated"
                             cif_summary += f"* **{idx}**: {sig_txt} with the cumulative incidence of the event (SHR={hr:.2f}, 95% CI {low:.2f}-{high:.2f}, p={p:.4f}).\n"
                     
                     st.success("Summary Generated:")
                     st.text_area("Copy this text:", value=cif_summary, height=200)

    # --- TAB 4: BIOMARKER DISCOVERY ---
    if 'tab4' in locals() and df_clean is not None:
         with tab4:
             st.header("Biomarker Cutoff Optimization")
             st.write("Evaluate valid continuous variables (e.g., Gene Expression, Lab Values) and find the optimal cutoff for survival stratification.")
             
             # Identify continuous columns (numeric & > 10 unique values)
             cont_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c]) and df_clean[c].nunique() > 10]
             # Exclude time/event/id columns if possible
             exclude = [time_col, event_col, "id", "ID", "patient_id"]
             cont_cols = [c for c in cont_cols if c not in exclude]
             
             if not cont_cols:
                 st.warning("No continuous variables found (columns with >10 unique numeric values).")
             else:
                 bio_col = st.selectbox("Select Biomarker (Continuous Variable)", cont_cols)
                 
                 col1, col2 = st.columns(2)
                 
                 with col1:
                     st.subheader("1. Continuous Association")
                     # Run Univariable Cox PH
                     try:
                         # normalize for better convergence
                         cox_df = df_clean[[time_col, event_col, bio_col]].replace([np.inf, -np.inf], np.nan).dropna()
                         
                         cph_cont = CoxPHFitter()
                         cph_cont.fit(cox_df, duration_col=time_col, event_col=event_col)
                         
                         summ = cph_cont.summary.loc[bio_col]
                         hr = summ['exp(coef)']
                         p_val = summ['p']
                         
                         st.metric("Hazard Ratio (Continuous)", f"{hr:.3f}", delta=None)
                         st.metric("P-value (Wald Test)", f"{p_val:.4f}", delta_color="inverse" if p_val < 0.05 else "normal")
                         
                         if p_val < 0.05:
                             st.success(f"**{bio_col}** is significantly associated with outcome as a continuous variable.")
                         else:
                             st.info(f"**{bio_col}** is NOT strictly associated as a linear continuous variable. A threshold effect might still exist.")
                             
                     except Exception as e:
                         # import might be missing if we used scikit-learn without importing.
                         # But actually CoxPHFitter is lifelines.
                         st.error(f"Analysis failed: {e}")

                     # --- ROC Section ---
                     st.divider()
                     st.subheader("3. Time-Dependent ROC Analysis")
                     st.write("Find optimal cutoff to predict event at a specific time point.")
                     
                     try:
                        # --- Pure NumPy Implementation of ROC/AUC to avoid dependency hell ---
                        def calculate_roc_auc_numpy(y_true, y_score):
                            # y_true: binary 0/1, y_score: continuous
                            
                            # Sort by score descending
                            desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
                            y_score = y_score[desc_score_indices]
                            y_true = y_true[desc_score_indices]
                            
                            # Distinct thresholds
                            distinct_value_indices = np.where(np.diff(y_score))[0]
                            threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
                            
                            # Accumulate positives (tps) and negatives (fps)
                            tps = np.cumsum(y_true)[threshold_idxs]
                            fps = (1 + threshold_idxs) - tps
                            
                            # Rates
                            tpr = tps / tps[-1]
                            fpr = fps / fps[-1]
                            
                            # Add 0,0 start point
                            tpr = np.r_[0, tpr]
                            fpr = np.r_[0, fpr]
                            thresholds = np.r_[y_score[0] + 1, y_score[threshold_idxs]]
                            
                            
                            # AUC using Trapezoidal rule (Version Agnostic for NumPy 1.x and 2.x)
                            # np.trapz was removed in NumPy 2.0
                            # Formula: sum( (y[i] + y[i-1]) * (x[i] - x[i-1]) ) / 2
                            # Note: fpr is increasing.
                            direction = 1
                            if fpr[0] > fpr[-1]:
                                direction = -1
                                
                            roc_auc = direction * np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr)) / 2
                            
                            return fpr, tpr, thresholds, roc_auc

                     except Exception as e:
                         st.error(f"Internal Function Error: {e}") 

                     roc_time = st.number_input("Target Time for Prediction (e.g., 24 months)", min_value=1.0, value=24.0, step=6.0)
                     
                     if st.button("Run ROC Analysis"):
                         # Prepare data for ROC
                         # Status at time t:
                         # 1 if Event=1 AND Time <= t
                         # 0 if Time > t
                         # Exclude if Event=0 AND Time <= t (Censored before t - unknown status)
                         
                         roc_data = df_clean[[time_col, event_col, bio_col]].replace([np.inf, -np.inf], np.nan).dropna()
                         
                         # Vectorized Filter
                         mask_event = (roc_data[event_col] == 1) & (roc_data[time_col] <= roc_time)
                         mask_no_event = (roc_data[time_col] > roc_time)
                         
                         roc_valid = roc_data[mask_event | mask_no_event].copy()
                         roc_valid['target'] = mask_event.astype(int)
                         
                         if len(roc_valid) < 10 or roc_valid['target'].nunique() < 2:
                             st.warning("Not enough valid data points (events and non-events) at this time point for ROC analysis.")
                         else:
                             try:
                                 # Calculate ROC (Custom NumPy)
                                 fpr, tpr, thresholds, roc_auc = calculate_roc_auc_numpy(roc_valid['target'].values, roc_valid[bio_col].values)
                                 
                                 # Find Optimal Cutoff (Youden's Index = TPR - FPR)
                                 # TPR = Sensitivity, FPR = 1 - Specificity
                                 youden = tpr - fpr
                                 best_idx = np.argmax(youden)
                                 best_thresh_roc = thresholds[best_idx]
                                 best_sens = tpr[best_idx]
                                 best_spec = 1 - fpr[best_idx]
                                 
                                 # NEW: Calculate Log-Rank P-value for this specific ROC-derived cutoff
                                 # This answers "Does this ROC cutoff actually separate survival curves?"
                                 group_roc = (cox_df[bio_col] > best_thresh_roc).astype(int)
                                 roc_p_val = np.nan
                                 try:
                                     if group_roc.sum() > 0 and (len(group_roc) - group_roc.sum()) > 0:
                                         res_roc = multivariate_logrank_test(cox_df[time_col], group_roc, cox_df[event_col])
                                         roc_p_val = res_roc.p_value
                                 except:
                                     pass
                                 
                                 st.success(f"**Optimal ROC Cutoff:** {best_thresh_roc:.2f} (AUC = {roc_auc:.3f})")
                                 
                                 col_roc_stats1, col_roc_stats2 = st.columns(2)
                                 with col_roc_stats1:
                                     st.write(f"Sensitivity: {best_sens:.2f}")
                                     st.write(f"Specificity: {best_spec:.2f}")
                                 with col_roc_stats2:
                                      if not np.isnan(roc_p_val):
                                          st.metric("Survival Separation P-value", f"{roc_p_val:.5f}", help="P-value from Log-Rank test using this cutoff.")
                                      else:
                                          st.write("P-value: N/A")
                                 
                                 # Plot ROC
                                 fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                                 ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                                 ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                 ax_roc.scatter(fpr[best_idx], tpr[best_idx], color='red', marker='o', s=100, label=f"Best: {best_thresh_roc:.2f}")
                                 ax_roc.set_xlim([0.0, 1.0])
                                 ax_roc.set_ylim([0.0, 1.05])
                                 ax_roc.set_xlabel('False Positive Rate')
                                 ax_roc.set_ylabel('True Positive Rate')
                                 ax_roc.set_title(f'ROC Curve at t={roc_time}')
                                 ax_roc.legend(loc="lower right")
                                 st.pyplot(fig_roc)
                                 
                                 # Store for Action
                                 st.session_state.optimal_cut = {
                                     'col': bio_col,
                                     'cut': best_thresh_roc,
                                     'p': 0.0 # Placeholder
                                 }
                                 
                             except Exception as e:
                                 st.error(f"ROC Calculation Error: {e}")
                 
                 with col2:
                     st.subheader("2. Cutoff Finder")
                     st.write("Test multiple cutoffs to maximize the Log-Rank difference (Lowest P-value).")
                     
                     if st.button(f"Find Optimal Cutoff for {bio_col}"):
                         # Calculate optimal cutoff
                         # Grid search between 10th and 90th percentile
                         values = cox_df[bio_col].sort_values()
                         low_p = np.percentile(values, 10)
                         high_p = np.percentile(values, 90)
                         
                         # Check 20 quantile steps
                         candidates = np.linspace(low_p, high_p, 50)
                         best_p = 1.0
                         best_cut = None
                         best_stats = None
                         
                         p_values = []
                         stats = []
                         
                         progress_bar = st.progress(0)
                         
                         for idx, cut in enumerate(candidates):
                             # Create temporary group
                             group_temp = (cox_df[bio_col] > cut).astype(int)
                             # Only test if we have events in both groups
                             if group_temp.sum() > 5 and (len(group_temp) - group_temp.sum()) > 5:
                                 res = multivariate_logrank_test(cox_df[time_col], group_temp, cox_df[event_col])
                                 p = res.p_value
                                 p_values.append(p)
                                 stats.append(res.test_statistic)
                                 
                                 if p < best_p:
                                     best_p = p
                                     best_cut = cut
                             else:
                                 p_values.append(np.nan)
                             
                             progress_bar.progress((idx + 1) / len(candidates))
                             
                         progress_bar.empty()
                         
                         if best_cut is not None:
                             st.success(f"**Optimal Cutoff Found:** {best_cut:.2f} (p = {best_p:.5f})")
                             
                             # Store in session state for plotting and saving
                             st.session_state.optimal_cut = {
                                 'col': bio_col,
                                 'cut': best_cut,
                                 'p': best_p
                             }
                             
                             # Plot P-value landscape
                             fig_p, ax_p = plt.subplots(figsize=(6, 4))
                             ax_p.plot(candidates, -np.log10(p_values), color='purple')
                             ax_p.set_ylabel("-Log10(P-value)")
                             ax_p.set_xlabel(f"{bio_col} Value")
                             ax_p.axvline(best_cut, color='red', linestyle='--', label=f"Best: {best_cut:.2f}")
                             ax_p.axhline(-np.log10(0.05), color='gray', linestyle=':', label="p=0.05")
                             ax_p.legend()
                             ax_p.set_title("Cutoff Optimization Landscape")
                             st.pyplot(fig_p)
                             
                 # 3. Action Section
                 st.divider()
                 st.subheader("3. Create Categorical Variable")
                 
                 # Default to optimal if found, else median
                 default_cut = df_clean[bio_col].median()
                 if 'optimal_cut' in st.session_state and st.session_state.optimal_cut['col'] == bio_col:
                     default_cut = st.session_state.optimal_cut['cut']
                     st.info(f"Optimal cutoff {default_cut:.2f} pre-filled from analysis above.")
                 
                 final_cut = st.number_input("Selected Cutoff Value", value=float(default_cut))
                 new_var_name = st.text_input("New Variable Name", value=f"{bio_col}_Cat")
                 
                 if st.button("Create & Add to Dataset"):
                     # Add to custom_cutoffs session state
                     new_def = {
                         'source': bio_col,
                         'value': final_cut,
                         'name': new_var_name
                     }
                     st.session_state.custom_cutoffs.append(new_def)
                     st.success(f"Variable **{new_var_name}** created! It is now available in the Main and CIF tabs.")
                     if hasattr(st, "rerun"):
                         st.rerun()
                     else:
                         st.experimental_rerun()

    # --- TAB 5: VARIABLE GENERATION ---
    if 'tab5' in locals() and df_clean is not None:
         with tab5:
             st.header("🧪 Variable Generation")
             st.write("Create advanced variables by combining existing columns or applying logical rules.")
             
             tab5_meth1, tab5_meth2 = st.tabs(["Method 1: Interaction Combiner", "Method 2: Boolean Logic"])
             
             # --- METHOD 1: INTERACTION COMBINER ---
             with tab5_meth1:
                 st.subheader("Combine Two Variables (Interaction)")
                 st.caption("Create unique groups for every combination of Variable A and Variable B (e.g., LSC+ & NGS+ -> 'LSC+_NGS+').")
                 
                 cols_avail = df_clean.columns.tolist()
                 # Exclude Time/Event if possible to keep lists clean, but user might want them.
                 
                 col1_gen, col2_gen = st.columns(2)
                 with col1_gen:
                     var_a = st.selectbox("Select Variable A", cols_avail, key="gen_var_a")
                 with col2_gen:
                     var_b = st.selectbox("Select Variable B", cols_avail, index=1 if len(cols_avail)>1 else 0, key="gen_var_b")
                 
                 if var_a == var_b:
                     st.warning("Please select two different variables.")
                 else:
                     # Preview
                     unique_a = df_clean[var_a].nunique()
                     unique_b = df_clean[var_b].nunique()
                     est_groups = unique_a * unique_b
                     
                     st.info(f"Variable A has **{unique_a}** levels. Variable B has **{unique_b}** levels. Result will have up to **{est_groups}** combined groups.")
                     
                     preview_name = st.text_input("New Variable Name (Method 1)", value=f"{var_a}_{var_b}_Combo")
                     
                     if st.button("Generate Combination Variable"):
                         # Add definition to session state
                         new_combo = {
                             'var1': var_a,
                             'var2': var_b,
                             'name': preview_name,
                             'type': 'interaction'
                         }
                         
                         if 'custom_combinations' not in st.session_state:
                             st.session_state.custom_combinations = []
                         
                         st.session_state.custom_combinations.append(new_combo)
                         st.success(f"Interaction variable **{preview_name}** created! It is now available in the sidebar.")
                         if hasattr(st, "rerun"):
                             st.rerun()
                         else:
                             st.experimental_rerun()
                             
             # --- METHOD 2: BOOLEAN LOGIC ---
             with tab5_meth2:
                 st.subheader("Create Group by Logic (Target vs Other)")
                 st.caption("Create a binary variable (Target / Other) based on specific conditions (e.g. Age > 60 AND LSC = Positive).")
                 
                 # Condition 1
                 c1_col1, c1_col2, c1_col3 = st.columns([2, 1, 2])
                 with c1_col1:
                     cond1_var = st.selectbox("Condition 1: Variable", cols_avail, key="blob_c1_var")
                 with c1_col2:
                     cond1_op = st.selectbox("Operator", [">", "<", ">=", "<=", "==", "!="], key="blob_c1_op")
                 with c1_col3:
                     # Try to detect type for input
                     if pd.api.types.is_numeric_dtype(df_clean[cond1_var]):
                         cond1_val = st.number_input("Value", value=0.0, key="blob_c1_val_num")
                     else:
                         vals_c1 = df_clean[cond1_var].unique()
                         cond1_val = st.selectbox("Value", vals_c1, key="blob_c1_val_str")
                         
                 # Logic Operator
                 logic_op = st.radio("Logic Operator", ["AND", "OR"], horizontal=True, key="blob_logic")
                 
                 # Condition 2
                 c2_col1, c2_col2, c2_col3 = st.columns([2, 1, 2])
                 with c2_col1:
                     cond2_var = st.selectbox("Condition 2: Variable", ["(None)"] + cols_avail, key="blob_c2_var")
                 
                 cond2_val = None
                 cond2_op = None
                 
                 if cond2_var != "(None)":
                     with c2_col2:
                         cond2_op = st.selectbox("Operator", [">", "<", ">=", "<=", "==", "!="], key="blob_c2_op")
                     with c2_col3:
                          if pd.api.types.is_numeric_dtype(df_clean[cond2_var]):
                             cond2_val = st.number_input("Value", value=0.0, key="blob_c2_val_num")
                          else:
                             vals_c2 = df_clean[cond2_var].unique()
                             cond2_val = st.selectbox("Value", vals_c2, key="blob_c2_val_str")
                             
                 new_bool_name = st.text_input("New Variable Name (Method 2)", value="Custom_Group")
                 
                 if st.button("Generate Boolean Variable"):
                     # We need to construct a definition that can be re-applied or applied now.
                     # Since boolean logic is complex to store purely as metadata without a parser, 
                     # we will execute it now and store the RESULTING COLUMN directly in the dataframe.
                     # However, to persist it, we need to re-run it every time?
                     # Ideally yes. For now, let's just modify the CURRENT df and rely on st.cache or session state persistence of the whole DF? 
                     # No, df is reloaded. We need to store the logic.
                     
                     bool_def = {
                         'type': 'boolean',
                         'name': new_bool_name,
                         'c1': {'var': cond1_var, 'op': cond1_op, 'val': cond1_val},
                         'logic': logic_op,
                         'c2': {'var': cond2_var, 'op': cond2_op, 'val': cond2_val} if cond2_var != "(None)" else None
                     }
                     
                     if 'custom_combinations' not in st.session_state:
                         st.session_state.custom_combinations = []
                     
                     st.session_state.custom_combinations.append(bool_def)
                     
                     # Force reload to apply logic in the main app loop (we need to update that loop too!)
                     st.success(f"Logic Variable **{new_bool_name}** created!")
                     if hasattr(st, "rerun"):
                         st.rerun()
                     else:
                         st.experimental_rerun()
                         
    # --- TAB 6: CORRELATIONS ---
    if 'tab6' in locals() and df_clean is not None:
         with tab6:
             st.header("🔥 Correlation Heatmap")
             st.write("Visualize relationships between variables. Useful for checking multicollinearity.")
             
             # 1. Select Columns
             all_cols_corr = df_clean.columns.tolist()
             numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
             
             # Default to numeric columns
             corr_cols = st.multiselect("Select Columns to Correlate", all_cols_corr, default=numeric_cols)
             
             if len(corr_cols) < 2:
                 st.warning("Please select at least 2 columns.")
             else:
                 # Correlation Settings
                 corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
                 
                 # Prepare Data
                 corr_df = df_clean[corr_cols].copy()
                 
                 # Encode Categoricals if selected
                 # Identify non-numeric
                 non_num_cols = [c for c in corr_cols if c not in numeric_cols]
                 if non_num_cols:
                     st.info(f"Encoding categorical columns: {', '.join(non_num_cols)}")
                     # Simple Factorization (Label Encoding) for correlation
                     # Or Get Dummies? For heatmap, label encoding creates a single 'axis' which is usually what user wants for "Is Group A correlated with B"
                     # Get dummies explodes the matrix too much.
                     for c in non_num_cols:
                         corr_df[c] = pd.factorize(corr_df[c])[0]
                 
                 # Compute Matrix
                 corr_matrix = corr_df.corr(method=corr_method)
                 
                 # Plot
                 fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                 
                 if sns is not None:
                     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr, square=True)
                     st.pyplot(fig_corr)
                 else:
                     st.error("Error: The 'seaborn' library could not be loaded.")
                     st.warning("The system attempted to auto-install it but failed. Please try restarting the app or installing 'seaborn' manually in your environment.")
                     st.code("pip install seaborn")
             


else:
    st.info("Please upload a CSV or Excel file to begin analysis.")
    st.write("Demostration with Dummy Data:")
    st.write("You can download the demo dataset `dummy_clinical_data.csv` from the repository:")
    st.markdown("[📂 View Repository & Download Data](https://github.com/gauravchatnobel/Easysurv)")
    st.caption("Right-click the link and open in a new tab to find the CSV file.")

