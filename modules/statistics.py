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


def sanitize_name(name):
    """Make a single column/level name safe for lifelines/formula fitting.

    Equivalent to the historical inline idiom
    ``x.replace(' ', '_').replace('+', 'pos').replace('-', 'neg')``.
    """
    return str(name).replace(' ', '_').replace('+', 'pos').replace('-', 'neg')


def sanitize_columns(df):
    """Return a copy of *df* with all column names sanitized (input unchanged)."""
    out = df.copy()
    out.columns = [sanitize_name(c) for c in out.columns]
    return out


def encode_with_reference(df, cat_cols, refs, dtype=float):
    """One-hot encode categorical columns, dropping each column's reference level.

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list[str] — categorical columns to encode.
    refs : dict[str, str] — reference level per column (the dropped dummy).
    dtype : numeric dtype for the dummy columns (float avoids bool-related
        dtype-inference issues in some pandas/lifelines versions).

    Returns
    -------
    (encoded_df, dummy_cols) — the frame with cat_cols replaced by dummies,
    and the list of new dummy column names (pre-sanitization).
    """
    out = df.copy()
    out = out.drop(columns=[c for c in cat_cols if c in out.columns], errors='ignore')
    dummy_cols = []
    for col in cat_cols:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, dtype=dtype)
        ref_col = f"{col}_{refs.get(col, '')}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])
        dummy_cols.extend(dummies.columns.tolist())
        out = pd.concat([out, dummies], axis=1)
    return out, dummy_cols


def bootstrap_optimism_c_index(df, time_col, event_col, covariates, n_boot=200, seed=42):
    """Optimism-corrected Harrell's C-index via Harrell's enhanced bootstrap.

    apparent  = C of the model fit on the full data, evaluated on the full data.
    optimism  = mean over bootstrap samples of (C on the bootstrap sample
                − C of the same bootstrap model on the original data).
    corrected = apparent − optimism.

    Returns dict: {'apparent', 'optimism', 'corrected', 'n_boot_used'} or None.
    Ref: Harrell FE. Regression Modeling Strategies (2015); Steyerberg (2009).
    """
    from lifelines.utils import concordance_index

    d = df[[time_col, event_col] + covariates].dropna()
    d_enc = pd.get_dummies(d, drop_first=True, dtype=float)
    d_enc.columns = [sanitize_name(c) for c in d_enc.columns]
    t_col = sanitize_name(time_col)
    e_col = sanitize_name(event_col)
    feat = [c for c in d_enc.columns if c not in (t_col, e_col)]
    if len(d_enc) < 20 or not feat:
        return None

    def _c_on(model, data):
        # Higher predicted survival should rank with longer times -> negate partial hazard
        risk = model.predict_partial_hazard(data)
        return concordance_index(data[t_col], -risk, data[e_col])

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        try:
            cph = CoxPHFitter()
            cph.fit(d_enc, duration_col=t_col, event_col=e_col)
            apparent = _c_on(cph, d_enc)
        except Exception:
            return None

        rng = np.random.RandomState(seed)
        optimisms = []
        n = len(d_enc)
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            boot = d_enc.iloc[idx]
            try:
                cph_b = CoxPHFitter()
                cph_b.fit(boot, duration_col=t_col, event_col=e_col)
                c_boot = _c_on(cph_b, boot)
                c_orig = _c_on(cph_b, d_enc)
                optimisms.append(c_boot - c_orig)
            except Exception:
                continue

    if len(optimisms) < max(10, n_boot // 4):
        return None
    optimism = float(np.mean(optimisms))
    return {
        'apparent': float(apparent),
        'optimism': optimism,
        'corrected': float(apparent - optimism),
        'n_boot_used': len(optimisms),
    }


def compute_calibration(df, time_col, event_col, covariates, horizon, n_bins=5, seed=42):
    """Calibration of a Cox model at a fixed time horizon.

    Patients are grouped by predicted survival at `horizon` into `n_bins`
    quantile bins. For each bin the mean predicted survival is compared with
    the Kaplan-Meier observed survival at `horizon` (with a 95% CI).

    Returns (calibration_df, meta) or (None, reason).
    calibration_df columns: Bin, n, Mean Predicted, Observed (KM), Obs Lower, Obs Upper.
    """
    d = df[[time_col, event_col] + covariates].dropna()
    d_enc = pd.get_dummies(d, drop_first=True, dtype=float)
    d_enc.columns = [sanitize_name(c) for c in d_enc.columns]
    t_col = sanitize_name(time_col)
    e_col = sanitize_name(event_col)
    feat = [c for c in d_enc.columns if c not in (t_col, e_col)]
    if len(d_enc) < 4 * n_bins or not feat:
        return None, "Not enough data for the requested number of bins."

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        try:
            cph = CoxPHFitter()
            cph.fit(d_enc, duration_col=t_col, event_col=e_col)
            surv = cph.predict_survival_function(d_enc, times=[horizon])
            pred = np.asarray(surv.iloc[0].values, dtype=float)  # S(horizon) per patient
        except Exception as e:
            return None, str(e)

    work = d_enc[[t_col, e_col]].copy()
    work['pred'] = pred
    # Quantile bins on predicted survival (unique edges to avoid empty bins)
    try:
        work['bin'] = pd.qcut(work['pred'].rank(method='first'), n_bins, labels=False)
    except Exception:
        return None, "Could not form calibration bins (too few distinct predictions)."

    rows = []
    for b in sorted(work['bin'].dropna().unique()):
        grp = work[work['bin'] == b]
        if len(grp) < 2:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(grp[t_col], grp[e_col])
        try:
            obs = float(kmf.predict(horizon))
            ci = kmf.confidence_interval_survival_function_
            # nearest index <= horizon
            idx = ci.index[ci.index <= horizon]
            if len(idx):
                lo = float(ci.loc[idx[-1]].iloc[0]); hi = float(ci.loc[idx[-1]].iloc[1])
            else:
                lo = hi = obs
        except Exception:
            obs = float('nan'); lo = hi = float('nan')
        rows.append({
            'Bin': int(b) + 1,
            'n': len(grp),
            'Mean Predicted': float(grp['pred'].mean()),
            'Observed (KM)': obs,
            'Obs Lower': lo,
            'Obs Upper': hi,
        })
    if not rows:
        return None, "No usable calibration bins."
    return pd.DataFrame(rows), {'horizon': horizon, 'n_bins': len(rows)}


def median_followup(times, events):
    """Median follow-up via the reverse Kaplan-Meier estimator (Schemper & Smith, 1996).

    Censoring is treated as the 'event', so the median is the time by which half
    the cohort would still be under observation. Returns a float in the same time
    units, or None if it cannot be estimated (e.g. not reached).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    mask = ~(np.isnan(times) | np.isnan(events))
    times, events = times[mask], events[mask]
    if len(times) == 0:
        return None
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(times, 1 - events)  # flip: censoring becomes the 'event'
        m = kmf.median_survival_time_
    except Exception:
        return None
    if m is None or np.isinf(m) or np.isnan(m):
        return None
    return float(m)


def adjust_pvalues(pvals, method="Benjamini-Hochberg"):
    """Adjust a list of p-values for multiple comparisons.

    Parameters
    ----------
    pvals : array-like of float (may contain NaN, which is passed through)
    method : "Benjamini-Hochberg" (FDR), "Bonferroni", or "None".

    Returns
    -------
    np.ndarray of adjusted p-values (same order as input), NaN preserved.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = pv.size
    if m == 0:
        return out
    if method in (None, "None", "none"):
        out[valid] = pv
        return out
    if method == "Bonferroni":
        out[valid] = np.minimum(pv * m, 1.0)
        return out
    # Benjamini-Hochberg (FDR): sort ascending, adjust, enforce monotonicity
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(m) + 1)
    # step-up: ensure non-decreasing from the largest p downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    result = np.empty(m, dtype=float)
    result[order] = adj
    out[valid] = result
    return out

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


def grays_test(df, time_col, event_col, group_col, event_of_interest=1, rho=0.0):
    """
    Gray's K-sample test for comparing cumulative incidence functions.

    This is a faithful port of the Fortran ``crst`` routine in R's
    ``cmprsk`` package (Robert Gray), so its output matches
    ``cmprsk::cuminc()$Tests`` to numerical precision. It is the
    competing-risks analogue of the log-rank test and tests the null
    hypothesis that the CIF of the event of interest is equal across
    all groups.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
    event_col : str
        0 = censored, ``event_of_interest`` = primary event, any other
        non-zero value = competing event.
    group_col : str
    event_of_interest : int
    rho : float
        Power in the Fleming-Harrington-type weight ``(1 - F(t))**rho``
        (cmprsk default 0, i.e. unweighted).

    Returns
    -------
    dict with keys: 'statistic', 'p_value', 'df'

    References
    ----------
    Gray RJ. A class of K-sample tests for comparing the cumulative
    incidence of a competing risk. Ann Stat 1988;16:1141-1154.
    Validated against cmprsk 2.2-12 (crst.f).
    """
    from scipy.stats import chi2

    df_clean = df[[time_col, event_col, group_col]].dropna().copy()
    groups = sorted(df_clean[group_col].unique())
    ng = len(groups)
    if ng < 2:
        return {'statistic': np.nan, 'p_value': np.nan, 'df': 0}

    group_idx = {g: i for i, g in enumerate(groups)}
    ig = df_clean[group_col].map(group_idx).to_numpy()
    ev = df_clean[event_col].to_numpy()
    # Recode to crst convention: 0 censored, 1 cause of interest, 2 competing
    m = np.where(ev == 0, 0, np.where(ev == event_of_interest, 1, 2)).astype(int)
    y = df_clean[time_col].to_numpy(dtype=float)

    stat, p_val, dfree = _grays_crst(y, m, ig, ng, rho)
    return {'statistic': stat, 'p_value': p_val, 'df': dfree}


def _grays_crst(y, m, ig, ng, rho=0.0):
    """Core of Gray's test — direct translation of cmprsk's crst.f (single stratum).

    y: failure times (sorted internally); m: 0=censored, 1=cause of interest,
    2=competing; ig: group index 0..ng-1. Returns (statistic, p_value, df).
    """
    from scipy.stats import chi2

    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=int)
    ig = np.asarray(ig, dtype=int)
    order = np.argsort(y, kind='mergesort')
    y, m, ig = y[order], m[order], ig[order]
    n = len(y)
    ng1 = ng - 1
    if ng1 < 1:
        return np.nan, np.nan, 0

    rs = np.zeros(ng)          # risk set size per group (ordinary)
    for j in ig:
        rs[j] += 1
    f1m = np.zeros(ng)         # CIF (cause 1), left-continuous
    f1 = np.zeros(ng)          # CIF (cause 1), right-continuous
    skmm = np.ones(ng)         # overall KM survival, left-continuous
    skm = np.ones(ng)          # overall KM survival, right-continuous
    v3 = np.zeros(ng)
    c = np.zeros((ng, ng))     # persistent across times
    v2 = np.zeros((ng1, ng))   # persistent across times
    a = np.zeros((ng, ng))
    V = np.zeros((ng1, ng1))   # variance, lower triangle accumulated
    s = np.zeros(ng1)          # score vector
    fm = 0.0
    f = 0.0

    ll = 0
    while ll < n:
        lu = ll
        while lu + 1 < n and y[lu + 1] == y[ll]:
            lu += 1

        d = np.zeros((3, ng))
        for i in range(ll, lu + 1):
            d[m[i], ig[i]] += 1
        nd1 = d[1].sum()
        nd2 = d[2].sum()

        if nd1 == 0 and nd2 == 0:
            # censoring-only time: reduce risk sets, leave S/F unchanged
            for i in range(ll, lu + 1):
                rs[ig[i]] -= 1
            ll = lu + 1
            continue

        tr = 0.0
        tq = 0.0
        for i in range(ng):
            if rs[i] <= 0:
                continue
            td = d[1, i] + d[2, i]
            skm[i] = skmm[i] * (rs[i] - td) / rs[i]
            f1[i] = f1m[i] + (skmm[i] * d[1, i]) / rs[i]
            tr += rs[i] / skmm[i]
            tq += rs[i] * (1 - f1m[i]) / skmm[i]

        f = fm + nd1 / tr
        fb = (1 - fm) ** rho

        a[:] = 0.0
        for i in range(ng):
            if rs[i] <= 0:
                continue
            t1 = rs[i] / skmm[i]
            a[i, i] = fb * t1 * (1 - t1 / tr)
            if a[i, i] != 0:
                c[i, i] += a[i, i] * nd1 / (tr * (1 - fm))
            for j in range(i + 1, ng):
                if rs[j] <= 0:
                    continue
                a[i, j] = -fb * t1 * rs[j] / (skmm[j] * tr)
                if a[i, j] != 0:
                    c[i, j] += a[i, j] * nd1 / (tr * (1 - fm))
        for i in range(ng):
            for j in range(i):
                a[i, j] = a[j, i]
                c[i, j] = c[j, i]

        for i in range(ng1):
            if rs[i] <= 0:
                continue
            s[i] += fb * (d[1, i] - nd1 * rs[i] * (1 - f1m[i]) / (skmm[i] * tq))

        if nd1 > 0:
            for k in range(ng):
                if rs[k] <= 0:
                    continue
                t4 = 1.0
                if skm[k] > 0:
                    t4 = 1 - (1 - f) / skm[k]
                t5 = 1.0
                if nd1 > 1:
                    t5 = 1 - (nd1 - 1) / (tr * skmm[k] - 1)
                t3 = t5 * skmm[k] * nd1 / (tr * rs[k])
                v3[k] += t4 * t4 * t3
                for i in range(ng1):
                    t1 = a[i, k] - t4 * c[i, k]
                    v2[i, k] += t1 * t4 * t3
                    for j in range(i + 1):
                        t2 = a[j, k] - t4 * c[j, k]
                        V[i, j] += t1 * t2 * t3

        if nd2 > 0:
            for k in range(ng):
                if skm[k] <= 0 or d[2, k] <= 0:
                    continue
                t4 = (1 - f) / skm[k]
                t5 = 1.0
                if d[2, k] > 1:
                    t5 = 1 - (d[2, k] - 1.0) / (rs[k] - 1.0)
                t3 = t5 * ((skmm[k] ** 2) * d[2, k]) / (rs[k] ** 2)
                v3[k] += t4 * t4 * t3
                for i in range(ng1):
                    t1 = t4 * c[i, k]
                    v2[i, k] -= t1 * t4 * t3
                    for j in range(i + 1):
                        t2 = t4 * c[j, k]
                        V[i, j] += t1 * t2 * t3

        if lu >= n - 1:
            break
        for i in range(ll, lu + 1):
            rs[ig[i]] -= 1
        fm = f
        f1m[:] = f1
        skmm[:] = skm
        ll = lu + 1

    for i in range(ng1):
        for j in range(i + 1):
            for k in range(ng):
                V[i, j] += c[i, k] * c[j, k] * v3[k]
                V[i, j] += c[i, k] * v2[j, k]
                V[i, j] += c[j, k] * v2[i, k]
    for i in range(ng1):
        for j in range(i):
            V[j, i] = V[i, j]

    try:
        stat = float(s @ np.linalg.inv(V) @ s)
        p_val = float(1 - chi2.cdf(stat, ng1))
    except np.linalg.LinAlgError:
        return np.nan, np.nan, ng1
    return stat, p_val, ng1


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
            
            # Fit weighted Cox for the subdistribution hazard. The correct
            # Fine-Gray variance is the cluster-robust (sandwich) estimator
            # clustered on subject id — passing cluster_col already forces
            # the sandwich estimator, so robust=True is stated explicitly to
            # keep intent and behaviour aligned. (The naive Hessian on the
            # IPCW-expanded data is far too small because it treats each
            # subject's pseudo-rows as independent.)
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                cph_pair = CoxPHFitter()
                cph_pair.fit(
                    fg_pair[cols_to_fit],
                    duration_col='stop', entry_col='start',
                    event_col='status', weights_col='weight',
                    cluster_col='id', robust=True
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
    d_enc.columns = [sanitize_name(c) for c in d_enc.columns]
    
    # Fit Main
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    try:
        cph.fit(d_enc, duration_col=time_col, event_col=event_col)
        c_est = cph.concordance_index_
    except Exception:
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
        except Exception:
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
            except Exception:
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
    except Exception:
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


def compute_rmst(df, time_col, event_col, group_col, tau):
    """
    Compute Restricted Mean Survival Time (RMST) per group.
    
    Uses analytical Greenwood-based variance — exact match to R's 
    survRM2::rmst2() (Uno et al., Statistics in Medicine 2014).
    
    RMST(τ) = ∫₀^τ S(t) dt = area under KM curve up to time τ.
    
    Variance formula:
        Var(RMST) = Σ [ψᵢ² × dᵢ / (nᵢ × (nᵢ − dᵢ))]
        where ψᵢ = ∫_{tᵢ}^τ S(u) du  (remaining area from tᵢ to τ)
    
    For differences: Var(diff) = Var(RMST₁) + Var(RMST₂)  (independent groups)
    P-value: z = diff / SE(diff), two-sided normal.
    """
    from itertools import combinations
    
    groups = sorted(df[group_col].dropna().unique())
    group_results = []
    
    for grp in groups:
        gdf = df[df[group_col] == grp].copy()
        
        # Administrative censoring at τ (equivalent to R's survRM2 logic)
        times = gdf[time_col].values.astype(float).copy()
        events = gdf[event_col].values.astype(float).copy()
        events[times > tau] = 0  # Censor at τ
        times = np.minimum(times, tau)
        
        # Fit KM
        kmf = KaplanMeierFitter()
        kmf.fit(times, events)
        
        # Event table — only event times (observed > 0) up to τ
        et = kmf.event_table
        event_mask = (et['observed'] > 0) & (et.index <= tau)
        et_events = et[event_mask]
        
        wk_time = et_events.index.values.astype(float)
        wk_n_risk = et_events['at_risk'].values.astype(float)
        wk_n_event = et_events['observed'].values.astype(float)
        
        # KM survival at event times
        wk_surv = kmf.survival_function_at_times(wk_time).values
        
        if len(wk_time) > 0:
            # ── RMST = area under KM step function [0, τ] ──
            # S(t)=1 for [0,t₁), S(t₁) for [t₁,t₂), ..., S(tₖ) for [tₖ,τ)
            all_times = np.concatenate([[0], wk_time, [tau]])
            all_surv = np.concatenate([[1], wk_surv])
            widths = np.diff(all_times)
            rmst_est = float(np.sum(all_surv * widths))
            
            # ── Variance: Greenwood formula (matches survRM2::rmst1) ──
            # ψᵢ = remaining area from tᵢ to τ
            intervals = np.diff(np.concatenate([wk_time, [tau]]))
            psi = np.flip(np.cumsum(np.flip(wk_surv * intervals)))
            
            denom = wk_n_risk * (wk_n_risk - wk_n_event)
            greenwood = np.where(denom > 0, wk_n_event / denom, 0)
            rmst_var = float(np.sum(greenwood * psi**2))
        else:
            rmst_est = float(tau)
            rmst_var = 0.0
        
        rmst_se = np.sqrt(rmst_var)
        
        group_results.append({
            'group': grp, 'rmst': rmst_est, 'se': rmst_se,
            'var': rmst_var,
            'lower': rmst_est - 1.96 * rmst_se,
            'upper': rmst_est + 1.96 * rmst_se,
            'n': len(gdf)
        })
    
    # Pairwise differences — analytical SE from independent variances
    pairwise = []
    for i, j in combinations(range(len(groups)), 2):
        g_i, g_j = groups[i], groups[j]
        diff_est = group_results[i]['rmst'] - group_results[j]['rmst']
        diff_var = group_results[i]['var'] + group_results[j]['var']
        diff_se = np.sqrt(diff_var)
        z = abs(diff_est / diff_se) if diff_se > 0 else 0
        p_val = 2 * (1 - norm.cdf(z))
        
        pairwise.append({
            'group_a': g_i, 'group_b': g_j,
            'diff': diff_est, 'se': diff_se,
            'lower': diff_est - 1.96 * diff_se,
            'upper': diff_est + 1.96 * diff_se,
            'p_value': p_val
        })
    
    difference = pairwise[0] if len(pairwise) == 1 else None
    
    return {
        'group_results': group_results,
        'difference': difference,
        'pairwise': pairwise,
        'tau': tau
    }


def compute_rmtl(df, time_col, event_col, group_col, event_of_interest, tau, n_boot=500):
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
        aj = AalenJohansenFitter(calculate_variance=False, seed=42)
        aj.fit(data[time_c], data[event_c], event_of_interest=eoi)
        # Build timeline from 0 to tau
        timeline = np.linspace(0, tau_val, 500)
        cif = aj.predict(timeline)
        return np.trapezoid(cif.values, timeline)
    
    # Per-group RMTL with bootstrap
    _rng = np.random.RandomState(42)  # Isolated RNG for reproducibility
    group_results = []
    group_rmtl_boots = {}
    
    for grp in groups:
        gdf = df[df[group_col] == grp].copy()
        try:
            rmtl_est = _rmtl_from_aj(gdf, time_col, event_col, event_of_interest, tau)
        except Exception:
            rmtl_est = 0
        
        # Bootstrap with explicit RNG (deterministic across reruns)
        boot_vals = []
        for _ in range(n_boot):
            boot_df = gdf.sample(n=len(gdf), replace=True, random_state=_rng)
            try:
                boot_vals.append(_rmtl_from_aj(boot_df, time_col, event_col, event_of_interest, tau))
            except Exception:
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
