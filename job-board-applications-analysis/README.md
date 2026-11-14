# Job board applications analysis (Jul–Sep 2026)
Run: open analysis.ipynb in Jupyter and Restart & Run All (needs pandas numpy matplotlib), or `python analysis.py job-board-applications.csv`
(fixed random seed 42, so numbers are identical every run). Outputs go to `output/`.

Files: analysis.ipynb (Jupyter notebook, executed with outputs), analysis.py (same work as a script), note_to_manager.md, output/chart_q3_reply_speed.png (the chart),
output/*.csv (result tables), output/cleaning_log.csv, output/cleaned.csv, output/run_log.txt.

Methods: Wilson 95% intervals for rates; permutation test (2,000 shuffles) for source vs hire;
bootstrap CI for the shortlisting gap. Consistency checks (hired⊂interviewed⊂shortlisted,
shortlisted⇒replied) all pass.
