import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.stats import norm

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


def _cache_data(func):
    """Apply st.cache_data if Streamlit is available, otherwise no-op."""
    if _HAS_STREAMLIT:
        return st.cache_data(show_spinner=False)(func)
    return func

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


def pairwise_fine_gray(df, time_col, event_col, group_col, event_of_interest=1):
    """
    Performs pairwise Fine-Gray regression between ALL pairs of groups.
    
    For each pair, fits a separate Fine-Gray model (IPCW-weighted Cox) using 
    only the two groups being compared. This is more powerful than extracting 
    individual HRs from the global model because:
      1. Each test uses only 1 degree of freedom (vs k-1 in the global model)
      2. The IPCW weights are computed on the pair-specific subset, giving 
         cleaner censoring estimates
    
    Returns a DataFrame with columns:
      Group 1, Group 2, Subdist HR, Lower 95%, Upper 95%, p-value
    
    Ref: Fine JP, Gray RJ. JASA 1999;94(446):496-509.
    """
    from itertools import combinations
    
    groups = sorted(df[group_col].dropna().unique())
    if len(groups) < 2:
        return None
    
    results = []
    
    for g1, g2 in combinations(groups, 2):
        try:
            # Subset to just these two groups
            pair_df = df[df[group_col].isin([g1, g2])].copy()
            
            # Compute Fine-Gray weights on the pair subset
            fg_pair = compute_fine_gray_weights(pair_df, time_col, event_col, event_of_interest)
            
            if fg_pair.empty or len(fg_pair) < 5:
                results.append({
                    'Group 1': str(g1), 'Group 2': str(g2),
                    'Subdist HR': np.nan, 'Lower 95%': np.nan,
                    'Upper 95%': np.nan, 'p-value': np.nan,
                    'Note': 'Insufficient data'
                })
                continue
            
            # Encode group: g2 = 1 (test), g1 = 0 (reference)
            fg_pair['_group_indicator'] = (fg_pair[group_col] == g2).astype(int)
            
            cols_to_fit = ['start', 'stop', 'status', 'weight', 'id', '_group_indicator']
            
            # Fit weighted Cox
            cph_pair = CoxPHFitter()
            cph_pair.fit(
                fg_pair[cols_to_fit],
                duration_col='stop', entry_col='start',
                event_col='status', weights_col='weight',
                cluster_col='id', robust=True
            )
            
            hr = cph_pair.summary.loc['_group_indicator', 'exp(coef)']
            lo = cph_pair.summary.loc['_group_indicator', 'exp(coef) lower 95%']
            hi = cph_pair.summary.loc['_group_indicator', 'exp(coef) upper 95%']
            p = cph_pair.summary.loc['_group_indicator', 'p']
            
            results.append({
                'Group 1': str(g1),
                'Group 2': str(g2),
                'Subdist HR': hr,
                'Lower 95%': lo,
                'Upper 95%': hi,
                'p-value': p,
            })
            
        except Exception as e:
            results.append({
                'Group 1': str(g1), 'Group 2': str(g2),
                'Subdist HR': np.nan, 'Lower 95%': np.nan,
                'Upper 95%': np.nan, 'p-value': np.nan,
                'Note': str(e)
            })
    
    if not results:
        return None
    
    return pd.DataFrame(results)

def calculate_wilson_ci(k, n, alpha=0.95):
    """Returns (lower, upper) tuple for Wilson Score Interval"""
    if n == 0: return 0.0, 0.0
    p = k / n
    z = norm.ppf(1 - (1 - alpha) / 2)
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z**2 / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z**2 / (4*n)) / n)
    lower = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return lower, upper

def get_c_index_bootstrap(df, time_col, event_col, covariates, label="", n_boot=50, penalizer=0.0, l1_ratio=0.0):
    """
    Fits Cox model and perform Bootstrap validation for C-Index estimation.
    Returns dictionary with result.
    """
    if not covariates: return None
    
    # Data Prep
    d = df[[time_col, event_col] + covariates].dropna()
    d_enc = pd.get_dummies(d, drop_first=True)
    d_enc.columns = [c.replace(' ', '_').replace('+', 'pos').replace('-', 'neg') for c in d_enc.columns]
    
    # Fit Main
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    try:
        cph.fit(d_enc, duration_col=time_col, event_col=event_col)
        c_est = cph.concordance_index_
    except:
        return None

    # Bootstrap (Normal Approximation Method)
    boot_cs = []
    for _ in range(n_boot):
        # Resample
        d_boot = d_enc.sample(n=len(d_enc), replace=True)
        try:
            cph_b = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            cph_b.fit(d_boot, duration_col=time_col, event_col=event_col)
            boot_cs.append(cph_b.concordance_index_)
        except:
            pass 
    
    if len(boot_cs) > 5:
        se = np.std(boot_cs)
        # 95% CI = Estimate +/- 1.96 * SE
        lower = max(0.0, c_est - 1.96 * se)
        upper = min(1.0, c_est + 1.96 * se)
    else:
        lower, upper = c_est, c_est
        
    return {"Label": label, "C-Index": c_est, "Lower": lower, "Upper": upper, "Vars": len(covariates)}

def check_epv(df, event_col, covariates):
    """
    Checks Events Per Variable (EPV) ratio using effective Degrees of Freedom.
    DoF = (Levels - 1) for categorical, 1 for numeric.
    Returns dict: {'status': 'green'/'yellow'/'red', 'message': str, 'value': float}
    """
    if not covariates: return {'status': 'green', 'message': 'No covariates selected.', 'value': float('inf')}
    
    n_events = df[event_col].sum()
    
    # Calculate Degrees of Freedom (DoF)
    n_params = 0
    df_cov = df[covariates].dropna()
    sparse_warnings = []
    
    for col in covariates:
        if pd.api.types.is_numeric_dtype(df_cov[col]) and len(df_cov[col].unique()) > 2:
            # Continuous variable = 1 DoF
            n_params += 1
        else:
            # Categorical or Binary
            unique_vals = len(df_cov[col].unique())
            # DoF is levels - 1 (e.g., 3 levels -> 2 dummy vars)
            # If unique_vals is 1 (constant), DoF is 0
            dof = max(0, unique_vals - 1)
            n_params += dof
            
            # Check for Sparse Events (Gap Check)
            try:
                # Group by level and sum events
                min_events = df.groupby(col)[event_col].sum().min()
                if min_events < 5:
                    sparse_warnings.append(f"⚠️ Categories in **{col}** have very few events (min={min_events}). Consider enabling **Penalized Cox** in Advanced Options.")
            except:
                pass
            
    epv = n_events / n_params if n_params > 0 else 0
    
    result = {'value': epv, 'sparse_warnings': sparse_warnings}
    
    if epv >= 15:
        result.update({'status': 'green', 'message': f"EPV = {epv:.1f} (Robust: {int(n_events)} events / {n_params} parameters)"})
    elif epv >= 10:
        result.update({'status': 'yellow', 'message': f"EPV = {epv:.1f} (Caution: {int(n_events)} events / {n_params} parameters)"})
    else:
        result.update({'status': 'red', 'message': f"EPV = {epv:.1f} (High Risk: {int(n_events)} events / {n_params} parameters)"})
        
    return result

@_cache_data
def get_correlation_matrix(df, covariates):
    """
    Returns the One-Hot Encoded correlation matrix for visualization.
    """
    if len(covariates) < 2: return None
    try:
         # dtype=int ensures we get 0/1 instead of True/False
         df_check = pd.get_dummies(df[covariates], drop_first=True, dtype=int).dropna()
         
         # Select numeric (int/float)
         df_check = df_check.select_dtypes(include=[np.number])
         
         if df_check.shape[1] < 2: return None
         return df_check.corr()
    except Exception as e:
         return None

@_cache_data
def calculate_vif(df, covariates):
    """
    Calculates Variance Inflation Factor (VIF) for covariates.
    Uses the diagonal of the inverse correlation matrix.
    Returns DataFrame: ['Feature', 'VIF']
    """
    if len(covariates) < 2: return None
    try:
        corr_matrix = get_correlation_matrix(df, covariates)
        if corr_matrix is None: return None
        
        # VIF is diagonal of inverse correlation matrix
        try:
            inv_corr = np.linalg.inv(corr_matrix.values)
        except np.linalg.LinAlgError:
            return None # Singular matrix (perfect collinearity)
            
        vif_values = np.diag(inv_corr)
        
        return pd.DataFrame({
            "Feature": corr_matrix.columns,
            "VIF": vif_values
        }).sort_values(by="VIF", ascending=False)
    except:
        return None

def check_collinearity(df, covariates, threshold=0.7):
    """
    Checks for multicollinearity using One-Hot Encoded correlations.
    Handles Numeric AND Categorical variables.
    Returns list of tuples: [('Var1', 'Var2', correlation)]
    """
    if len(covariates) < 2: return []
    
    corr_matrix = get_correlation_matrix(df, covariates)
    if corr_matrix is None: return []
    
    # Upper triangle only
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr = []
    for col in upper.columns:
        for row in upper.index:
             val = upper.loc[row, col]
             if abs(val) > threshold:
                 high_corr.append((row, col, float(val)))
                 
    return high_corr

def check_separation(cph_model):
    """
    Checks a fitted paramaters for signs of complete separation.
    Returns list of warnings.
    """
    warnings = []
    
    # 1. Check for infinite/huge coefficients (perfect predictor)
    # Threshold |beta| > 10 is usually a sign (HR > 20,000 or < 0.00005)
    high_coefs = cph_model.params_[abs(cph_model.params_) > 10].index.tolist()
    if high_coefs:
        warnings.append(f"Potential Separation: Extreme coefficients found for {', '.join(high_coefs)}.")
        
    # 2. Check for huge Standard Errors (SE > 10)
    # This often happens when variance is infinite
    high_se = cph_model.standard_errors_[cph_model.standard_errors_ > 5].index.tolist()
    if high_se:
        warnings.append(f"Unstable Estimates: Large Standard Errors found for {', '.join(high_se)}.")
        
    return warnings

def summarize_model_risk(epv_res, collinearity_list, vif_df, separation_warnings):
    """
    Synthesizes multiple statistical checks into a unified model health score.
    Returns dict: {
        'status': 'green'/'yellow'/'red',
        'label': 'Robust'/'Caution'/'High Risk',
        'reasons': [str],
        'recommendation': str
    }
    """
    reasons = []
    status = 'green'
    
    # Check 1: EPV
    if epv_res['status'] == 'red':
        status = 'red'
        reasons.append(f"❌ Critical Sample Size Issue: {epv_res['message']}")
    elif epv_res['status'] == 'yellow':
        if status != 'red': status = 'yellow'
        reasons.append(f"⚠️ Low Sample Size: {epv_res['message']}")
        
    # Check 1b: Sparse Events
    if epv_res.get('sparse_warnings'):
        if status != 'red': status = 'yellow'
        reasons.append("⚠️ Sparse Events detected in specific subgroups.")

    # Check 2: Separation
    if separation_warnings:
        status = 'red'
        reasons.append("❌ Complete Separation Detected (Infinite HRs).")
        
    # Check 3: VIF
    if vif_df is not None:
        max_vif = vif_df['VIF'].max()
        if max_vif > 10:
            status = 'red'
            reasons.append(f"❌ Severe Multicollinearity (Max VIF={max_vif:.1f}).")
        elif max_vif > 5:
            if status != 'red': status = 'yellow'
            reasons.append(f"⚠️ Potential Multicollinearity (Max VIF={max_vif:.1f}).")
            
    # Check 4: Raw Correlation (Backup)
    if collinearity_list and (vif_df is None or vif_df.empty):
         if status != 'red': status = 'yellow'
         reasons.append(f"⚠️ High Correlation detected ({len(collinearity_list)} pairs > 0.7).")

    # Final Interpretation
    rec = "Model appears statistically robust."
    label = "Robust"
    
    if status == 'red':
        label = "High Risk"
        rec = "Results are likely unreliable. Consider reducing variables, simplifying categorical levels, or using Penalized Cox."
    elif status == 'yellow':
        label = "Caution"
        rec = "Results interpretability may be limited. Proceed with care."
        
    return {
        'status': status,
        'label': label,
        'reasons': reasons,
        'recommendation': rec
    }
