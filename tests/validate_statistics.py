#!/usr/bin/env python3
"""Quick statistical validation runner for EasySurv."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from modules.statistics import (
    calculate_wilson_ci, compute_fine_gray_weights, check_epv,
    calculate_vif, check_separation, get_c_index_bootstrap, summarize_model_risk,
)
from scipy.stats import norm

results = []
def chk(name, cond):
    results.append((name, cond))
    print(f'  {"✅" if cond else "❌"} {name}')

print("=== 1. KAPLAN-MEIER ===")
kmf = KaplanMeierFitter()
kmf.fit([1,2,3,4,5,6,7,8,9,10], [1,1,1,1,1,1,1,1,1,1])
chk("S(1)=0.9", abs(kmf.survival_function_at_times(1).iloc[0]-0.9)<0.01)
chk("S(5)=0.5", abs(kmf.survival_function_at_times(5).iloc[0]-0.5)<0.01)
chk("Median in [5,6]", 5<=kmf.median_survival_time_<=6)

# Censoring should increase survival estimate
kmf_c = KaplanMeierFitter()
kmf_c.fit([1,2,3,4,5,6,7,8,9,10], [1,0,1,0,1,0,1,0,1,0])
chk("Censoring inflates S(t)", kmf_c.survival_function_at_times(5).iloc[0] >= kmf.survival_function_at_times(5).iloc[0])

print("\n=== 2. COX PH ===")
np.random.seed(42)
d = pd.DataFrame({
    "Time": np.concatenate([np.random.exponential(30,100), np.random.exponential(15,100)]),
    "Event": np.concatenate([np.random.binomial(1,0.65,100), np.random.binomial(1,0.75,100)]),
    "GroupB": np.array([0]*100+[1]*100),
    "Age": np.random.normal(65,10,200),
})
cph = CoxPHFitter()
cph.fit(d[["Time","Event","GroupB"]], duration_col="Time", event_col="Event")
hr = cph.summary.loc["GroupB","exp(coef)"]
lo = cph.summary.loc["GroupB","exp(coef) lower 95%"]
hi = cph.summary.loc["GroupB","exp(coef) upper 95%"]
p = cph.summary.loc["GroupB","p"]
chk(f"HR(B vs A)={hr:.2f} > 1 (B worse)", hr>1)
chk(f"95% CI [{lo:.2f}-{hi:.2f}] contains HR", lo<=hr<=hi)
chk(f"p={p:.4f} < 0.05", p<0.05)
chk(f"C-Index={cph.concordance_index_:.3f} > 0.5", cph.concordance_index_>0.5)

print("\n=== 3. PENALIZED COX ===")
pen = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
pen.fit(d[["Time","Event","GroupB"]], duration_col="Time", event_col="Event")
chk("Penalized shrinks |β|", abs(pen.params_["GroupB"]) <= abs(cph.params_["GroupB"]))

print("\n=== 4. LOG-RANK TEST ===")
g = np.array(["A"]*100+["B"]*100)
lr = multivariate_logrank_test(d["Time"], g, d["Event"])
chk(f"Different groups: p={lr.p_value:.4f} < 0.05", lr.p_value<0.05)
np.random.seed(42)
lr2 = multivariate_logrank_test(
    np.random.exponential(20,200),
    np.array(["A"]*100+["B"]*100),
    np.random.binomial(1,0.5,200),
)
chk(f"Same groups: p={lr2.p_value:.3f} > 0.05 (no false positive)", lr2.p_value>0.05)

print("\n=== 5. PH ASSUMPTION TEST ===")
ph = proportional_hazard_test(cph, d[["Time","Event","GroupB"]], time_transform="rank")
chk("PH p-value in [0,1]", 0 <= ph.summary["p"].values[0] <= 1)

print("\n=== 6. WILSON CI ===")
lo_w, hi_w = calculate_wilson_ci(7, 10)
z = norm.ppf(0.975); p_w = 0.7
denom = 1+z**2/10; c = p_w+z**2/20; s = np.sqrt((p_w*0.3+z**2/40)/10)
exp_lo = (c-z*s)/denom; exp_hi = (c+z*s)/denom
chk("Wilson matches mathematical formula", abs(lo_w-exp_lo)<1e-10 and abs(hi_w-exp_hi)<1e-10)
chk("Wilson(0,0) = (0,0)", calculate_wilson_ci(0,0) == (0.0, 0.0))
lo_b, hi_b = calculate_wilson_ci(5, 10)
chk("Lower <= Upper always", lo_b <= hi_b)

# Coverage simulation
np.random.seed(42)
covered = sum(1 for _ in range(500) if calculate_wilson_ci(np.random.binomial(50,0.3),50)[0]<=0.3<=calculate_wilson_ci(np.random.binomial(50,0.3),50)[1])
chk(f"Wilson ~95% coverage ({covered/5:.0f}%)", 85 < covered/5 < 100)

print("\n=== 7. FINE-GRAY COMPETING RISKS ===")
np.random.seed(123)
cr = pd.DataFrame({
    "Time": np.abs(np.random.exponential(20,150))+0.01,
    "Event": np.random.choice([0,1,2], 150, p=[0.25,0.45,0.30]),
    "Group": np.random.choice(["X","Y"], 150),
})
fg = compute_fine_gray_weights(cr, "Time", "Event", event_of_interest=1)
chk("Has start/stop/status/weight columns", all(c in fg.columns for c in ["start","stop","status","weight"]))
chk("All weights > 0", (fg["weight"]>0).all())
chk(f"Start weights ≈ n ({fg[fg['start']==0]['weight'].sum():.0f}≈150)", abs(fg[fg["start"]==0]["weight"].sum()-150)<1)
chk("Event subjects weight=1", (fg[(fg["start"]==0)&(fg["status"]==1)]["weight"]==1.0).all())

decay_ok = True
for pid in fg["id"].unique()[:20]:
    rows = fg[fg["id"]==pid].sort_values("start")
    if len(rows)>1:
        w = rows["weight"].values
        if not all(w[i]<=w[i-1]+1e-10 for i in range(1,len(w))):
            decay_ok = False; break
chk("Competing risk weights decay", decay_ok)

# FG Cox fit
fg2 = pd.get_dummies(fg, columns=["Group"], drop_first=True)
gc = [c for c in fg2.columns if c.startswith("Group_")]
try:
    cph_fg = CoxPHFitter()
    cph_fg.fit(fg2[["start","stop","status","weight","id"]+gc],
               duration_col="stop", entry_col="start",
               event_col="status", weights_col="weight",
               cluster_col="id", robust=True)
    chk(f"FG Cox fit: HR={cph_fg.summary['exp(coef)'].values[0]:.3f}", True)
except Exception as e:
    chk(f"FG Cox fit failed: {e}", False)

print("\n=== 8. AALEN-JOHANSEN CIF ===")
ajf = AalenJohansenFitter()
ajf.fit(cr["Time"], cr["Event"], event_of_interest=1)
cif = ajf.cumulative_density_
chk("CIF bounded [0,1]", (cif.values>=-1e-10).all() and (cif.values<=1+1e-10).all())
chk("CIF monotonically non-decreasing", all(d>=-1e-10 for d in np.diff(cif.values.flatten())))

ajf2 = AalenJohansenFitter()
ajf2.fit(cr["Time"], cr["Event"], event_of_interest=2)
cm = cif.index.intersection(ajf2.cumulative_density_.index)
ms = max(cif.loc[t].iloc[0]+ajf2.cumulative_density_.loc[t].iloc[0] for t in cm)
chk(f"Sum of CIFs max={ms:.3f} <= 1.0", ms<=1+1e-10)

print("\n=== 9. C-INDEX BOOTSTRAP ===")
r = get_c_index_bootstrap(d, "Time", "Event", ["GroupB"], label="Test", n_boot=20)
chk(f"C-Index={r['C-Index']:.3f} in [0,1]", 0<=r["C-Index"]<=1)
chk("Lower <= Est <= Upper", r["Lower"]<=r["C-Index"]<=r["Upper"])
r_none = get_c_index_bootstrap(d, "Time", "Event", [], n_boot=10)
chk("Empty covariates -> None", r_none is None)

print("\n=== 10. EPV ===")
epv = check_epv(d, "Event", ["GroupB","Age"])
ne = d["Event"].sum(); exp_epv = ne/2
chk(f"EPV={epv['value']:.1f} ≈ {exp_epv:.1f}", abs(epv["value"]-exp_epv)<1)
chk("Green status (EPV>15)", epv["status"]=="green")

print("\n=== 11. VIF ===")
np.random.seed(42)
idf = pd.DataFrame({"X1":np.random.normal(0,1,500),"X2":np.random.normal(0,1,500),"X3":np.random.normal(0,1,500)})
v = calculate_vif(idf, ["X1","X2","X3"])
chk(f"Indep VIF max={v['VIF'].max():.2f} < 2", v["VIF"].max()<2)

x = np.random.normal(0,1,100)
cdf = pd.DataFrame({"X1":x, "X2":x+np.random.normal(0,0.01,100)})
v2 = calculate_vif(cdf, ["X1","X2"])
chk(f"Collinear VIF={v2['VIF'].max():.0f} >> 10", v2["VIF"].max()>100)

print("\n=== 12. SEPARATION DETECTION ===")
chk("Normal model: 0 warnings", len(check_separation(cph))==0)

print("\n=== 13. MODEL RISK SUMMARY ===")
r1 = summarize_model_risk(epv, [], None, [])
chk("Clean → Green/Robust", r1["status"]=="green" and r1["label"]=="Robust")
r2 = summarize_model_risk({"status":"red","message":"EPV=3","value":3}, [], None, ["Sep found"])
chk("Bad → Red/High Risk", r2["status"]=="red" and r2["label"]=="High Risk")

vif_bad = pd.DataFrame({"Feature":["X1"], "VIF":[15.0]})
r3 = summarize_model_risk({"status":"green","message":"","value":30}, [], vif_bad, [])
chk("High VIF → Red", r3["status"]=="red")

print("\n" + "="*60)
passed = sum(1 for _,c in results if c)
total = len(results)
print(f"RESULT: {passed}/{total} checks passed ({100*passed/total:.0f}%)")
if passed == total:
    print("🎉 ALL STATISTICAL CHECKS PASSED")
else:
    print("FAILURES:")
    for n,c in results:
        if not c: print(f"  ❌ {n}")
