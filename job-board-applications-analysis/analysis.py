"""Job board applications, Jul-Sep 2026. Run: python analysis.py job-board-applications.csv
Writes cleaned data, result tables and chart into ./output"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "job-board-applications.csv"
OUT = "output"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- 1. Clean ----------
raw = pd.read_csv(SRC, keep_default_na=False)
log = {"rows_in": len(raw)}
df = raw.drop_duplicates().copy()
log["exact_duplicate_rows_removed"] = len(raw) - len(df)
assert df.application_id.is_unique
df["work_mode"] = df.work_mode.str.lower().str.replace("-", "_")
log["work_mode_case_fixed"] = int((raw.work_mode != raw.work_mode.str.lower().str.replace("-", "_")).sum())
df["source"] = df.source.replace("", "unknown")
log["source_blank_set_unknown"] = int((df.source == "unknown").sum())
for c in ["shortlisted", "interviewed", "hired"]:
    df[c] = (df[c] == "yes").astype(int)
df["reply_days"] = pd.to_numeric(df.days_to_first_reply, errors="coerce")
df["replied"] = df.reply_days.notna().astype(int)
df["paid"] = df.stipend_inr_month > 0
df["all4"] = df.steps_done == 4
df["applied_on"] = pd.to_datetime(df.applied_on)
# consistency checks (all should be 0)
log["hired_without_interview"] = int((df.hired & ~df.interviewed.astype(bool)).sum())
log["interview_without_shortlist"] = int((df.interviewed & ~df.shortlisted.astype(bool)).sum())
log["shortlisted_without_reply"] = int(((df.shortlisted == 1) & (df.replied == 0)).sum())
log["rows_out"] = len(df)
df.drop(columns="days_to_first_reply").to_csv(f"{OUT}/cleaned.csv", index=False)
pd.Series(log).to_csv(f"{OUT}/cleaning_log.csv", header=["value"])

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h

# ---------- Q1 ----------
def per100(g):
    n, k = len(g), g.hired.sum(); lo, hi = wilson(k, n)
    return pd.Series({"applications": n, "hires": k, "hires_per_100": 100*k/n,
                      "ci_low": 100*lo, "ci_high": 100*hi})
q1 = df.groupby("source").apply(per100, include_groups=False).sort_values("hires_per_100", ascending=False)
q1.round(2).to_csv(f"{OUT}/q1_source_overall.csv")
q1r = df.groupby(["role_category", "source"]).apply(per100, include_groups=False).reset_index()
q1r.round(2).to_csv(f"{OUT}/q1_source_by_role.csv", index=False)
piv = q1r.pivot(index="role_category", columns="source", values="hires_per_100").round(1)
piv.to_csv(f"{OUT}/q1_pivot_hires_per_100.csv")
# does referral win in every role? + permutation test of source effect
top = q1.index[0]
q1r["rank_in_role"] = q1r.groupby("role_category").hires_per_100.rank(ascending=False)
known = df[df.source != "unknown"]
def chi(d):
    t = pd.crosstab(d.source, d.hired); e = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.values.sum()
    return ((t.values - e)**2 / e).sum()
obs = chi(known); perm = [chi(known.assign(source=rng.permutation(known.source.values))) for _ in range(2000)]
q1_p = (np.sum(np.array(perm) >= obs) + 1) / 2001

# ---------- Q2 ----------
def rate(g): 
    n, k = len(g), g.shortlisted.sum(); lo, hi = wilson(k, n)
    return pd.Series({"applicants": n, "shortlisted": k, "shortlist_rate_pct": 100*k/n, "ci_low": 100*lo, "ci_high": 100*hi})
q2 = df.groupby("all4").apply(rate, include_groups=False)
q2.index = ["fewer than 4 steps", "all 4 steps"]; q2.round(2).to_csv(f"{OUT}/q2_all4_vs_not.csv")
q2s = df.groupby("steps_done").apply(rate, include_groups=False); q2s.round(2).to_csv(f"{OUT}/q2_by_steps.csv")
a, b = df[df.all4], df[~df.all4]
diff = 100*(a.shortlisted.mean() - b.shortlisted.mean())
boot = [100*(a.shortlisted.sample(len(a), replace=True, random_state=int(s)).mean() -
             b.shortlisted.sample(len(b), replace=True, random_state=int(s)+1).mean()) for s in rng.integers(0, 1e9, 2000)]
dlo, dhi = np.percentile(boot, [2.5, 97.5])
ratio = a.shortlisted.mean() / b.shortlisted.mean()
# robustness: within role, and within paid/unpaid
q2role = df.groupby(["role_category", "all4"]).shortlisted.mean().unstack()*100
q2role.columns = ["<4 steps", "4 steps"]; q2role.round(1).to_csv(f"{OUT}/q2_by_role.csv")
q2paid = df.groupby(["paid", "all4"]).shortlisted.mean().unstack()*100
q2paid.round(1).to_csv(f"{OUT}/q2_by_paid.csv")

# ---------- Q3 ----------
def reply(g):
    r = g.reply_days.dropna(); n, k = len(g), (g.replied == 0).sum(); lo, hi = wilson(k, n)
    return pd.Series({"applications": n, "no_reply_pct": 100*k/n, "no_reply_ci_low": 100*lo, "no_reply_ci_high": 100*hi,
                      "median_days": r.median(), "mean_days": r.mean(), "p75_days": r.quantile(.75), "p90_days": r.quantile(.9),
                      "within_3_days_pct_of_all": 100*(g.reply_days <= 3).mean()})
q3 = df.groupby("paid").apply(reply, include_groups=False); q3.index = ["unpaid", "paid"]
q3.round(2).to_csv(f"{OUT}/q3_paid_vs_unpaid.csv")
q3role = df.groupby(["role_category", "paid"]).apply(reply, include_groups=False)[["applications", "no_reply_pct", "median_days"]].round(1)
q3role.to_csv(f"{OUT}/q3_by_role.csv")
q3mode = df.groupby(["work_mode", "paid"]).apply(reply, include_groups=False)[["applications", "no_reply_pct", "median_days"]].round(1)
q3mode.to_csv(f"{OUT}/q3_by_workmode.csv")
q3stip = df.groupby("stipend_inr_month").apply(reply, include_groups=False)[["applications", "no_reply_pct", "median_days"]].round(1)
q3stip.to_csv(f"{OUT}/q3_by_stipend.csv")
pa, un = df[df.paid].reply_days.dropna(), df[~df.paid].reply_days.dropna()
mdiff = [np.median(rng.choice(un, len(un))) - np.median(rng.choice(pa, len(pa))) for _ in range(2000)]
# top-coded sensitivity: trim replies >14 days
tr = df[(df.reply_days.isna()) | (df.reply_days <= 14)]
q3_trim = tr.groupby("paid").reply_days.mean().round(2)
print("replies >14 days:", int((df.reply_days > 14).sum()), "| trimmed mean paid/unpaid:", q3_trim.to_dict())

# ---------- Chart (Q3) ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
bins = list(range(1, 11)) + [11]; lab = [str(i) for i in range(1, 10)] + ["10", "11+"]
cols = {"paid": "#1b6ca8", "unpaid": "#e07a1f"}
w = .38
for i, (name, m) in enumerate([("paid", df.paid), ("unpaid", ~df.paid)]):
    g = df[m]; s = g.reply_days.clip(upper=11).value_counts().reindex(range(1, 12), fill_value=0)
    ax[0].bar(np.arange(11) + (i-.5)*w, 100*s.values/len(g), w, color=cols[name], label=f"{name.capitalize()} (n={len(g)})")
ax[0].set_xticks(range(11)); ax[0].set_xticklabels(lab)
ax[0].set_xlabel("Days until first reply"); ax[0].set_ylabel("% of that group's applications")
ax[0].set_title("Paid roles reply faster", loc="left", fontweight="bold"); ax[0].legend(frameon=False)
nr = q3.no_reply_pct
for i, name in enumerate(["paid", "unpaid"]):
    lo, hi = q3.loc[name, ["no_reply_ci_low", "no_reply_ci_high"]]; v = nr[name]
    ax[1].bar(i, v, .55, color=cols[name], yerr=[[v-lo], [hi-v]], capsize=4)
    ax[1].text(i, hi+1, f"{v:.0f}%", ha="center", fontweight="bold")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels([f"Paid\nmedian {q3.loc['paid','median_days']:.0f} days", f"Unpaid\nmedian {q3.loc['unpaid','median_days']:.0f} days"])
ax[1].set_ylabel("% never replied"); ax[1].set_ylim(0, 42); ax[1].set_title("…and are less often ignored", loc="left", fontweight="bold")
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("Jul–Sep 2026: unpaid roles reply twice as slowly and are ignored 2x as often (error bars: 95% CI)", fontsize=10, y=1.0)
plt.tight_layout(); plt.savefig(f"{OUT}/chart_q3_reply_speed.png", dpi=160, bbox_inches="tight")

# ---------- Print summary ----------
pd.set_option("display.width", 200)
print(pd.Series(log).to_string()); print("\nQ1\n", q1.round(1)); print("perm p (source vs hire, known):", round(q1_p, 4))
print(piv); print(q1r.pivot(index="role_category", columns="source", values="applications"))
print("\nQ2\n", q2.round(1)); print(f"diff {diff:.1f} pts  CI [{dlo:.1f},{dhi:.1f}]  ratio {ratio:.2f}")
print(q2s.round(1)); print(q2role.round(1)); print(q2paid.round(1))
print("\nQ3\n", q3.round(1).T); print("median diff boot CI", np.percentile(mdiff, [2.5, 97.5]))
print(q3role); print(q3mode); print(q3stip)
