# Job board applications analysis (Jul–Sep 2026)

Analysis of 1,200 applications (made-up dataset) to answer three questions before a planning meeting.

## Key findings
| Question | Answer |
|---|---|
| 1. Best source for hires? | **Referral**: 10.7 hires per 100 applications, then email 8.1, search 5.7, college 3.4, social 2.3. It does not hold in every role (e.g. search leads in Data, email in Software), and role-level counts are tiny (≤6 hires per cell). |
| 2. Do applicants who finish all 4 steps get shortlisted more? | **Yes**: 48.4% vs 9.3%, a gap of about 39 points (95% CI 34–44), about 5×. Holds in every role and for paid and unpaid roles. |
| 3. Paid vs unpaid reply speed? | Median **3 days vs 7**; **13% vs 30%** never hear back. The gap shows in every role and work mode, so it is big enough to act on. |

**Recommendation:** nudge employers on unpaid roles to reply (reminder at day 3, flag at day 7).

## Data cleaning
- Removed 6 exact duplicate rows (1,206 → 1,200)
- Standardised 9 inconsistent `work_mode` labels (e.g. "On-site" → `on_site`)
- Set 15 blank `source` values to "unknown"
- Checked logic: hired ⊂ interviewed ⊂ shortlisted, and shortlisted ⇒ replied (all pass)

## Method
- 95% Wilson intervals for rates
- Permutation test (2,000 shuffles) for source vs hire
- Bootstrap CI for the shortlisting gap
- Fixed random seed (42), so results are identical on every run

## Caveats
Only 64 hires in total; a blank reply may mean "not yet" rather than "never"; a stipend of 0 may mean "not stated"; the step-completion result is a correlation, not proof of cause; the data is synthetic.

## Files
| File | What it is |
|---|---|
| `analysis.ipynb` | Main work, executed with outputs |
| `analysis.py` | Same analysis as a script |
| `note_to_manager.md` | ~200-word note with answers and recommendation |
| `output/chart_q3_reply_speed.png` | The chart (reply speed, paid vs unpaid) |
| `output/*.csv` | Result tables and cleaning log |
| `job-board-applications.csv` | Source data |

## Run it
```bash
pip install pandas numpy matplotlib jupyter
jupyter notebook analysis.ipynb      # Restart & Run All
# or
python analysis.py job-board-applications.csv
```

## AI disclosure
The code, chart and note were drafted with Claude (an AI assistant). Numbers can be verified by re-running the notebook.
