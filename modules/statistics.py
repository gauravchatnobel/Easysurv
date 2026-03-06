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


def grays_test(df, time_col, event_col, group_col, event_of_interest=1):
    """
    Gray's K-sample test for comparing cumulative incidence functions.
    
    This is the competing-risks analogue of the log-rank test, equivalent
    to R's cmprsk::cuminc()$Tests. It tests the null hypothesis that the 
    CIF of the event of interest is equal across all groups.
    
    Uses a modified weighted log-rank statistic where subjects with 
    competing events remain in the subdistribution risk set with IPCW
    weights G(t)/G(Ti).
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    event_col : str
        0 = censored, event_of_interest = primary event, other values = competing events.
    group_col : str
    event_of_interest : int
    
    Returns
    -------
    dict with keys: 'statistic', 'p_value', 'df'
    
    References
    ----------
    Gray RJ. A class of K-sample tests for comparing the cumulative 
    incidence of a competing risk. Ann Stat 1988;16:1141-1154.
    """
    from scipy.stats import chi2
    
    df_clean = df[[time_col, event_col, group_col]].dropna().copy()
    groups = sorted(df_clean[group_col].unique())
    K = len(groups)
    
    if K < 2:
        return {'statistic': np.nan, 'p_value': np.nan, 'df': 0}
    
    # ---- Step 1: Estimate censoring survival G(t) = P(C > t) ----
    # Reverse KM: "event" = censoring (event_col == 0)
    times_all = df_clean[time_col].values
    events_all = df_clean[event_col].values
    
    # Build censoring KM manually for efficiency
    cens_indicator = (events_all == 0).astype(int)
    unique_times = np.sort(np.unique(times_all))
    
    # At each time: n at risk, n censored (= "event" for G)
    G_values = {}
    n_remaining = len(times_all)
    G_current = 1.0
    
    for t in unique_times:
        at_t = times_all == t
        d_cens = int(cens_indicator[at_t].sum())  # censoring events
        n_events = int((~cens_indicator.astype(bool) & at_t).sum())  # real events
        
        if n_remaining > 0 and d_cens > 0:
            G_current *= (1 - d_cens / n_remaining)
        
        G_values[t] = max(G_current, 1e-10)
        n_remaining -= (d_cens + n_events)
    
    def G_at(t):
        """G(t-): censoring survival just before time t."""
        prev = 1.0
        for ut in unique_times:
            if ut >= t:
                return prev
            prev = G_values[ut]
        return prev
    
    # ---- Step 2: Identify event-of-interest times ----
    eoi_mask = events_all == event_of_interest
    eoi_times = np.sort(np.unique(times_all[eoi_mask]))
    
    if len(eoi_times) == 0:
        return {'statistic': np.nan, 'p_value': np.nan, 'df': K - 1}
    
    # ---- Step 3: Pre-compute per-subject info ----
    group_idx = {g: i for i, g in enumerate(groups)}
    subj_time = times_all
    subj_event = events_all
    subj_group = np.array([group_idx[g] for g in df_clean[group_col].values])
    
    # Identify competing event subjects
    is_competing = np.array([e != 0 and e != event_of_interest for e in subj_event])
    competing_times = subj_time[is_competing]
    competing_groups = subj_group[is_competing]
    competing_G_Ti = np.array([G_at(t) for t in competing_times])
    
    # ---- Step 4: Compute U and V ----
    U = np.zeros(K - 1)
    V = np.zeros((K - 1, K - 1))
    
    for t in eoi_times:
        G_t = G_at(t)
        
        # d_j(t): events of interest in each group at time t
        at_t_eoi = (subj_time == t) & (subj_event == event_of_interest)
        d = np.zeros(K)
        for j in range(K):
            d[j] = int(at_t_eoi[subj_group == j].sum())
        
        # R_j(t): subdistribution risk set for each group
        # = subjects with T_i >= t (still in study)
        # + competing event subjects with T_i < t, weighted by G(t)/G(T_i)
        still_at_risk = subj_time >= t
        R = np.zeros(K)
        for j in range(K):
            R[j] = int(still_at_risk[subj_group == j].sum())
        
        # Add IPCW contribution from competing events before t
        before_t = competing_times < t
        if before_t.any():
            weights = np.where(competing_G_Ti[before_t] > 1e-10,
                              G_t / competing_G_Ti[before_t], 0.0)
            for j in range(K):
                in_group = competing_groups[before_t] == j
                R[j] += weights[in_group].sum()
        
        d_total = d.sum()
        R_total = R.sum()
        
        if R_total < 1e-10 or d_total == 0:
            continue
        
        # Score: U_j += d_j - R_j * d/R
        for j in range(K - 1):
            U[j] += d[j] - R[j] * d_total / R_total
        
        # Variance: V_j1j2 = Σ_t R_j1*(delta_{j1j2}*R - R_j2) * d*(R-d) / (R^2*(R-1))
        # For j1==j2: R_j * (R - R_j) * d*(R-d) / (R^2*(R-1))
        # For j1!=j2: -R_j1 * R_j2 * d*(R-d) / (R^2*(R-1))
        if R_total > 1 and d_total < R_total:
            factor = d_total * (R_total - d_total) / (R_total * R_total * (R_total - 1))
            for j1 in range(K - 1):
                for j2 in range(K - 1):
                    if j1 == j2:
                        V[j1, j2] += R[j1] * (R_total - R[j1]) * factor
                    else:
                        V[j1, j2] -= R[j1] * R[j2] * factor
    
    # ---- Step 5: Test statistic ----
    try:
        V_inv = np.linalg.inv(V)
        stat = float(U @ V_inv @ U)
        p_val = 1 - chi2.cdf(stat, df=K - 1)
    except np.linalg.LinAlgError:
        stat = np.nan
        p_val = np.nan
    
    return {'statistic': stat, 'p_value': p_val, 'df': K - 1}


def pairwise_fine_gray(df, time_col, event_col, group_col, event_of_interest=1, reference_group=None):
    """
    Performs pairwise Fine-Gray regression between pairs of groups.
    
    For each pair, fits a separate Fine-Gray model (IPCW-weighted Cox) using 
    only the two groups being compared. This is more powerful than extracting 
    individual HRs from the global model because:
      1. Each test uses only 1 degree of freedom (vs k-1 in the global model)
      2. The IPCW weights are computed on the pair-specific subset, giving 
         cleaner censoring estimates
    
    Parameters
    ----------
    reference_group : str or None
        If provided, all comparisons use this group as the baseline (Group 1).
        HR > 1 means Group 2 has higher cumulative incidence than reference.
        If None, all pairwise combinations are shown with alphabetical ordering.
    
    Returns a DataFrame with columns:
      Reference, Comparison, Subdist HR, Lower 95%, Upper 95%, p-value
    
    Ref: Fine JP, Gray RJ. JASA 1999;94(446):496-509.
    """
    from itertools import combinations
    
    groups = sorted(df[group_col].dropna().unique())
    if len(groups) < 2:
        return None
    
    # Build list of (reference, comparison) pairs
    if reference_group is not None and str(reference_group) in [str(g) for g in groups]:
        # All other groups vs the chosen reference
        pairs = [(reference_group, g) for g in groups if str(g) != str(reference_group)]
    else:
        # All pairwise combinations (alphabetical order)
        pairs = list(combinations(groups, 2))
    
    results = []
    
    for ref, comp in pairs:
        try:
            # Subset to just these two groups
            pair_df = df[df[group_col].isin([ref, comp])].copy()
            
            # Compute Fine-Gray weights on the pair subset
            fg_pair = compute_fine_gray_weights(pair_df, time_col, event_col, event_of_interest)
            
            if fg_pair.empty or len(fg_pair) < 5:
                results.append({
                    'Reference': str(ref), 'Comparison': str(comp),
                    'Subdist HR': np.nan, 'Lower 95%': np.nan,
                    'Upper 95%': np.nan, 'p-value': np.nan,
                    'Note': 'Insufficient data'
                })
                continue
            
            # Encode group: comp = 1 (test), ref = 0 (baseline)
            fg_pair['_group_indicator'] = (fg_pair[group_col] == comp).astype(int)
            
            cols_to_fit = ['start', 'stop', 'status', 'weight', 'id', '_group_indicator']
            
            # Fit weighted Cox with model-based (Hessian) SE
            # This matches R's cmprsk::crr() variance estimator.
            # Do NOT use robust=True (sandwich SE is overly conservative).
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                cph_pair = CoxPHFitter()
                cph_pair.fit(
                    fg_pair[cols_to_fit],
                    duration_col='stop', entry_col='start',
                    event_col='status', weights_col='weight',
                    cluster_col='id', robust=False
                )
            
            from scipy.stats import norm as _norm
            _beta = cph_pair.params_.values[0]
            _se = np.sqrt(cph_pair.variance_matrix_.values[0, 0])
            hr = np.exp(_beta)
            lo = np.exp(_beta - 1.96 * _se)
            hi = np.exp(_beta + 1.96 * _se)
            p = 2 * (1 - _norm.cdf(abs(_beta / _se)))
            
            results.append({
                'Reference': str(ref),
                'Comparison': str(comp),
                'Subdist HR': hr,
                'Lower 95%': lo,
                'Upper 95%': hi,
                'p-value': p,
            })
            
        except Exception as e:
            results.append({
                'Reference': str(ref), 'Comparison': str(comp),
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


def compute_rmst(df, time_col, event_col, group_col, tau, n_boot=200):
    """
    Compute Restricted Mean Survival Time (RMST) per group.
    
    RMST(τ) = ∫₀^τ S(t) dt = area under KM curve up to time τ.
    
    Difference tested via bootstrap (group1 - group2).
    Equivalent to R's survRM2::rmst2().
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    event_col : str
    group_col : str
    tau : float
        Restriction time (must be ≤ min of max observed time per group).
    n_boot : int
        Number of bootstrap replicates.
    
    Returns
    -------
    dict with keys:
      'group_results': list of {group, rmst, se, lower, upper, n}
      'difference': {diff, se, lower, upper, p_value} or None (if >2 groups)
      'tau': float
    """
    groups = sorted(df[group_col].dropna().unique())
    
    def _rmst_single(data, time_c, event_c, tau_val):
        """Compute RMST for a single dataset."""
        kmf = KaplanMeierFitter()
        kmf.fit(data[time_c], data[event_c])
        # Get survival function up to tau
        sf = kmf.survival_function_at_times(np.sort(np.unique(np.append(
            data[time_c][data[time_c] <= tau_val].values, [0, tau_val]
        )))).values
        times = np.sort(np.unique(np.append(
            data[time_c][data[time_c] <= tau_val].values, [0, tau_val]
        )))
        # Trapezoidal integration
        return np.trapezoid(sf, times)
    
    def _rmst_from_kmf(data, time_c, event_c, tau_val):
        """More robust RMST using lifelines KMF timeline."""
        kmf = KaplanMeierFitter()
        kmf.fit(data[time_c], data[event_c])
        # Build timeline from 0 to tau
        timeline = np.linspace(0, tau_val, 500)
        sf = kmf.predict(timeline)
        return np.trapezoid(sf.values, timeline)
    
    # Per-group RMST with bootstrap
    group_results = []
    group_rmst_boots = {}
    
    for grp in groups:
        gdf = df[df[group_col] == grp].copy()
        rmst_est = _rmst_from_kmf(gdf, time_col, event_col, tau)
        
        # Bootstrap
        boot_vals = []
        for _ in range(n_boot):
            boot_df = gdf.sample(n=len(gdf), replace=True)
            try:
                boot_vals.append(_rmst_from_kmf(boot_df, time_col, event_col, tau))
            except:
                pass
        
        se = np.std(boot_vals) if len(boot_vals) > 5 else 0
        ci_low = max(0, rmst_est - 1.96 * se)
        ci_high = rmst_est + 1.96 * se
        
        group_results.append({
            'group': grp, 'rmst': rmst_est, 'se': se,
            'lower': ci_low, 'upper': ci_high, 'n': len(gdf)
        })
        group_rmst_boots[grp] = boot_vals
    
    # Pairwise differences (all pairs)
    pairwise = []
    from itertools import combinations
    for i, j in combinations(range(len(groups)), 2):
        g_i, g_j = groups[i], groups[j]
        diff_est = group_results[i]['rmst'] - group_results[j]['rmst']
        
        n_min = min(len(group_rmst_boots[g_i]), len(group_rmst_boots[g_j]))
        if n_min > 5:
            boot_diffs = [group_rmst_boots[g_i][k] - group_rmst_boots[g_j][k] for k in range(n_min)]
            diff_se = np.std(boot_diffs)
            diff_ci_low = diff_est - 1.96 * diff_se
            diff_ci_high = diff_est + 1.96 * diff_se
            z = abs(diff_est / diff_se) if diff_se > 0 else 0
            p_val = 2 * (1 - norm.cdf(z))
        else:
            diff_se = 0; diff_ci_low = diff_est; diff_ci_high = diff_est; p_val = 1.0
        
        pairwise.append({
            'group_a': g_i, 'group_b': g_j,
            'diff': diff_est, 'se': diff_se,
            'lower': diff_ci_low, 'upper': diff_ci_high,
            'p_value': p_val
        })
    
    # Backward compat: 'difference' = first pair if exactly 2 groups
    difference = pairwise[0] if len(pairwise) == 1 else None
    
    return {
        'group_results': group_results,
        'difference': difference,
        'pairwise': pairwise,
        'tau': tau
    }


def compute_rmtl(df, time_col, event_col, group_col, event_of_interest, tau, n_boot=200):
    """
    Compute Restricted Mean Time Lost (RMTL) per group from CIF.
    
    RMTL(τ) = ∫₀^τ CIF(t) dt = area under CIF curve up to time τ.
    Interpretable as "average time lost to the event within [0, τ]."
    
    Ref: Andersen PK. Stat Med 2013; Zhao et al. 2016.
    Equivalent to computing RMTL from R's tidycmprsk or adjustedCurves.
    
    Parameters
    ----------
    df : pd.DataFrame
    time_col, event_col, group_col : str
    event_of_interest : int/float
        Code for the event of interest in event_col.
    tau : float
        Restriction time.
    n_boot : int
        Bootstrap replicates.
    
    Returns
    -------
    dict with same structure as compute_rmst but for RMTL.
    """
    from lifelines import AalenJohansenFitter
    
    groups = sorted(df[group_col].dropna().unique())
    
    def _rmtl_from_aj(data, time_c, event_c, eoi, tau_val):
        """Compute RMTL from Aalen-Johansen CIF."""
        aj = AalenJohansenFitter(calculate_variance=False)
        aj.fit(data[time_c], data[event_c], event_of_interest=eoi)
        # Build timeline from 0 to tau
        timeline = np.linspace(0, tau_val, 500)
        cif = aj.predict(timeline)
        return np.trapezoid(cif.values, timeline)
    
    # Per-group RMTL with bootstrap
    group_results = []
    group_rmtl_boots = {}
    
    for grp in groups:
        gdf = df[df[group_col] == grp].copy()
        try:
            rmtl_est = _rmtl_from_aj(gdf, time_col, event_col, event_of_interest, tau)
        except:
            rmtl_est = 0
        
        # Bootstrap
        boot_vals = []
        for _ in range(n_boot):
            boot_df = gdf.sample(n=len(gdf), replace=True)
            try:
                boot_vals.append(_rmtl_from_aj(boot_df, time_col, event_col, event_of_interest, tau))
            except:
                pass
        
        se = np.std(boot_vals) if len(boot_vals) > 5 else 0
        ci_low = max(0, rmtl_est - 1.96 * se)
        ci_high = rmtl_est + 1.96 * se
        
        group_results.append({
            'group': grp, 'rmtl': rmtl_est, 'se': se,
            'lower': ci_low, 'upper': ci_high, 'n': len(gdf)
        })
        group_rmtl_boots[grp] = boot_vals
    
    # Pairwise differences (all pairs)
    pairwise = []
    from itertools import combinations
    for i, j in combinations(range(len(groups)), 2):
        g_i, g_j = groups[i], groups[j]
        diff_est = group_results[i]['rmtl'] - group_results[j]['rmtl']
        
        n_min = min(len(group_rmtl_boots[g_i]), len(group_rmtl_boots[g_j]))
        if n_min > 5:
            boot_diffs = [group_rmtl_boots[g_i][k] - group_rmtl_boots[g_j][k] for k in range(n_min)]
            diff_se = np.std(boot_diffs)
            diff_ci_low = diff_est - 1.96 * diff_se
            diff_ci_high = diff_est + 1.96 * diff_se
            z = abs(diff_est / diff_se) if diff_se > 0 else 0
            p_val = 2 * (1 - norm.cdf(z))
        else:
            diff_se = 0; diff_ci_low = diff_est; diff_ci_high = diff_est; p_val = 1.0
        
        pairwise.append({
            'group_a': g_i, 'group_b': g_j,
            'diff': diff_est, 'se': diff_se,
            'lower': diff_ci_low, 'upper': diff_ci_high,
            'p_value': p_val
        })
    
    # Backward compat: 'difference' = first pair if exactly 2 groups
    difference = pairwise[0] if len(pairwise) == 1 else None
    
    return {
        'group_results': group_results,
        'difference': difference,
        'pairwise': pairwise,
        'tau': tau
    }
