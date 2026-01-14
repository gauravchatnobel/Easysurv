
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
import numpy as np
import io

def add_at_risk_counts(*fitters, ax=None, y_shift=-0.25, colors=None, labels=None):
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
                ha='right', va='center', weight='bold', color=color, fontsize=10)
        
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
                    ha='center', va='center', color=color, fontsize=10)
    
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

if uploaded_file:
    # Load Data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Column Selection
    columns = df.columns.tolist()
    
    st.sidebar.subheader("Variable Selection")
    
    # Time and Event columns
    time_col = st.sidebar.selectbox("Time Column (Duration)", columns, index=columns.index("OS_Days") if "OS_Days" in columns else 0)
    event_col = st.sidebar.selectbox("Event Column (Status: 1=Event, 0=Censored)", columns, index=columns.index("OS_Status") if "OS_Status" in columns else 0)
    
    # Grouping Variable
    group_col = st.sidebar.selectbox("Grouping Variable (e.g., MRD Status)", ["None"] + columns, index=columns.index("MRD_Status") + 1 if "MRD_Status" in columns else 0)

    # Plot Settings
    st.sidebar.subheader("Plot Settings")
    show_risk_table = st.sidebar.checkbox("Show At-Risk Table", value=True)
    table_height = st.sidebar.slider("Risk Table Offset (Move Down)", -0.5, -0.1, -0.25, 0.05) if show_risk_table else -0.25
    show_censored = st.sidebar.checkbox("Show Censored Ticks", value=True)
    show_ci = st.sidebar.checkbox("Show 95% CI Shading", value=True)
    show_hr_plot = st.sidebar.checkbox("Show Hazard Ratio (HR) on Plot", value=False)
    
    # Fonts
    font_options = ["sans-serif", "serif", "monospace", "Arial", "Times New Roman", "Courier New", "Verdana"]
    selected_font = st.sidebar.selectbox("Plot Font", font_options, index=0)
    plt.rcParams['font.family'] = selected_font
    
    # Axes & Layout
    st.sidebar.subheader("Axes & Layout")
    x_label = st.sidebar.text_input("X-Axis Label", value="Time (Months)")
    tick_interval = st.sidebar.number_input("X-Axis Tick Interval", min_value=1, value=12, help="Set to 12 for yearly ticks if data is in months.")
    
    y_tick_interval = st.sidebar.number_input("Y-Axis Tick Interval", min_value=0.01, value=0.1, step=0.05)
    y_min = st.sidebar.number_input("Y-Axis Min", value=0.0, step=0.1)
    y_max = st.sidebar.number_input("Y-Axis Max", value=1.0, step=0.1)
    
    plot_height = st.sidebar.slider("Plot Height", 4, 12, 6)
    plot_width = st.sidebar.slider("Plot Width", 6, 15, 10)
    
    # Theme Selection
    st.sidebar.subheader("Aesthetics & Themes")

    # Theme Selection
    st.sidebar.subheader("Aesthetics & Themes")
    
    # Define Palettes
    journal_themes = {
        "Nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"],
        "Cell": ["#C0392B", "#2980B9", "#27AE60", "#F1C40F", "#8E44AD", "#7F8C8D", "#D35400"],
        "Science": ["#0C7BDC", "#E66100", "#5D3A9B", "#1A85FF", "#D41159", "#FFC20A"],
        "Blood": ["#882E72", "#B178A6", "#D6C1DE", "#1965B0", "#5289C7", "#7BAFDE", "#4EB265", "#CAE0AB"],
        "Journal of Clinical Oncology": ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6"],
        "Lancet": ["#00539C", "#EE352E", "#FF9933", "#009392", "#3fa68a", "#90e4c1"],
        "Leukemia": ["#377EB8", "#E41A1C", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF"],
        "Cancer Discovery": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
    }
    
    fun_themes = {
        "Rick and Morty": ["#A6EEE6", "#F0F035", "#44281D", "#E4A71B", "#8BCF21", "#FBFBFB"], # Portal Green, Morty Yellow, Rick Hair
        "Bojack Horseman": ["#2C3E50", "#D35400", "#2980B9", "#C0392B", "#bdc3c7", "#F39C12"],
        "Primal": ["#5D4037", "#D84315", "#388E3C", "#FBC02D", "#455A64", "#212121"],
        "Scavenger's Reign": ["#F4EBD0", "#D66853", "#3C505D", "#7D9D9C", "#212D40", "#A8C686"],
        "Common Side Effects": ["#FF00FF", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF0000"] # Neon chaos
    }
    
    all_themes = {**journal_themes, **fun_themes}
    theme_names = ["Default"] + list(journal_themes.keys()) + list(fun_themes.keys()) + ["Custom"]
    
    selected_theme = st.sidebar.selectbox("Choose Theme", theme_names)
    
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
            
    # Legend Renaming
    group_labels = {}
    if group_col != "None" and df_clean is not None:
        with st.sidebar.expander("📝 Rename Legend Labels"):
            unique_groups = sorted(df_clean[group_col].dropna().unique())
            for grp in unique_groups:
                new_label = st.text_input(f"Label for {grp}", value=str(grp))
                group_labels[grp] = new_label

    # Analysis
    if time_col and event_col and df_clean is not None:
        st.divider()
        st.header("Survival Analysis")

        tab1, tab2 = st.tabs(["Univariable Analysis (Kaplan-Meier)", "Multivariable Analysis (Cox PH)"])

        with tab1:
        
            if dropped_count > 0:
                st.warning(f"⚠️ **Missing Values Detected**: Dropped {dropped_count} rows containing NaNs in the selected columns. Analysis based on {len(df_clean)} remaining rows.")
                with st.expander("See details of missing values"):
                    st.write("Number of missing values per column (in the original selection):")
                    # Calculate missing per column in the subset
                    missing_stats = df[cols_to_check].isna().sum()
                    st.dataframe(missing_stats[missing_stats > 0])
            
            # Calculate HR for Plot if requested
            hr_text = ""
            if show_hr_plot and group_col != "None" and len(df_clean[group_col].unique()) >= 2:
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
                    hr_text = f"p = {res.p_value:.4f}"
                except:
                    pass

        
            # 1. Plotting
            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        
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
                    kmf_group.plot_survival_function(ax=ax, ci_show=show_ci, show_censors=show_censored, color=color)
                    fitters.append(kmf_group)
            
                if hr_text:
                    ax.text(0.7, 0.9, hr_text, transform=ax.transAxes, fontsize=12, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                # Risk Table logic (basic implementation using lifelines built-in if possible, or custom)
                if show_risk_table:
                    # Custom add_at_risk_counts integration
                    # We need fitters for all to use add_at_risk_counts
                    # fitters list already populated above
                    add_at_risk_counts(*fitters, ax=ax, y_shift=table_height, colors=plot_colors, labels=plot_labels)

                # Apply Custom Label
                ax.set_xlabel(x_label)
                st.pyplot(fig)

                # DOWNLOAD BUTTON
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="💾 Download High-Res Plot (300 DPI)",
                    data=buf,
                    file_name="survival_plot.png",
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
                    st.caption(f"Reference Group: **{reference_group}** (All HRs are relative to this group)")

                except Exception as e:
                    st.error(f"Cox Model Error: {e}")
                    st.info("Tip: This often happens if a group has too few events or perfectly predicts survival (separation).")
                
                except Exception as e:
                    st.warning(f"Could not run Cox PH model: {e}")

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
            
                st.table(pd.DataFrame(est_data))

            else:
                # Single group
                color = None
                if selected_theme in all_themes:
                     color = all_themes[selected_theme][0]
                elif selected_theme == "Custom":
                     color = st.sidebar.color_picker("Color for All Patients", "#1f77b4")

                if selected_theme == "Custom":
                     color = st.sidebar.color_picker("Color for All Patients", "#1f77b4")

                kmf_all = KaplanMeierFitter()
                kmf_all.fit(df_clean[time_col], df_clean[event_col], label="All Patients")
                kmf_all.plot_survival_function(ax=ax, ci_show=show_ci, show_censors=show_censored, color=color)
                if show_risk_table:
                    # from lifelines.plotting import add_at_risk_counts (REMOVED due to bug)
                    add_at_risk_counts(kmf_all, ax=ax)
            
                # Apply Custom Label
                ax.set_xlabel(x_label)
                st.pyplot(fig)

                # DOWNLOAD BUTTON
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="💾 Download High-Res Plot (300 DPI)",
                    data=buf,
                    file_name="survival_plot.png",
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
                    if st.button("Run Multivariable Analysis"):
                        try:
                            # Encore Categorical Variables
                            # Detect categorical columns (object or category)
                            # Actually, get_dummies handles this, but we should be careful about reference levels.
                            # get_dummies(drop_first=True) automatically drops one level to avoid multicollinearity.
                            
                            mv_data_encoded = pd.get_dummies(mv_df, columns=covariates, drop_first=True)
                            
                            # Sanitize Column Names for Lifelines/Stats (remove spaces/special chars)
                            # This is important for formula strings but CoxPHFitter handles dataframe input well.
                            # However, cleaner names are better for the plot.
                            mv_data_encoded.columns = [c.replace(' ', '_').replace('+', 'pos').replace('-', 'neg') for c in mv_data_encoded.columns]
                            
                            # Fit Model
                            cph_mv = CoxPHFitter()
                            cph_mv.fit(mv_data_encoded, duration_col=time_col, event_col=event_col)
                            
                            # Results Table
                            summary_mv = cph_mv.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
                            summary_mv.columns = ['Hazard Ratio (HR)', 'Lower 95% CI', 'Upper 95% CI', 'p-value']
                            
                            st.write("### Regression Results")
                            st.dataframe(summary_mv.style.format("{:.3f}"))
                            
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
                            # xerr expected as shape (2, N) : [[left_errs], [right_errs]]
                            # left_err = HR - Lower
                            # right_err = Upper - HR
                            x_errs = [
                                plot_data['Hazard Ratio (HR)'] - plot_data['Lower 95% CI'],
                                plot_data['Upper 95% CI'] - plot_data['Hazard Ratio (HR)']
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
                            
                            # Download
                            buf_forest = io.BytesIO()
                            fig_forest.savefig(buf_forest, format="png", dpi=300, bbox_inches='tight')
                            buf_forest.seek(0)
                            st.download_button("💾 Download Forest Plot", buf_forest, "forest_plot.png", "image/png")
                            
                        except Exception as e:
                            st.error(f"Error running model: {e}")
                            st.info("Ensure you are not including variables that perfectly predict the outcome (separation).")

            else:
                st.info("Select at least one covariate variable (e.g., Age, Gender, Mutations) to begin.")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")
    st.write("Demostration with Dummy Data:")
    st.write("You can download the demo dataset `dummy_clinical_data.csv` from the repository:")
    st.markdown("[📂 View Repository & Download Data](https://github.com/gauravchatnobel/Easysurv)")
    st.caption("Right-click the link and open in a new tab to find the CSV file.")
