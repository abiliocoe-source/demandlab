"""Builds and executes notebooks/demandlab.ipynb (thin: all logic lives in the demandlab package)."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# DemandLab: exploration and results\nDaily bike rentals, UCI Bike Sharing (2011-2012). "
       "Run `python -m demandlab.train` first; this notebook only explores and reads saved results."),
    code("import json, matplotlib.pyplot as plt, pandas as pd\n"
         "from demandlab.data import load_raw, get_series\n"
         "from demandlab.analysis import trend, weekday_profile, month_profile, detect_anomalies\n"
         "raw = load_raw(); y = get_series(raw, 'total'); y.describe().round(0)"),
    md("## Trend"),
    code("print(trend(y))\ny.plot(figsize=(10,3), title='Daily rentals (total)'); plt.show()"),
    md("## Seasonality (index vs overall mean; 1.0 = average)"),
    code("fig, ax = plt.subplots(1, 2, figsize=(10,3))\n"
         "weekday_profile(y).plot.bar(ax=ax[0], title='By weekday (0=Mon)')\n"
         "month_profile(y).plot.bar(ax=ax[1], title='By month'); plt.tight_layout(); plt.show()"),
    md("## Anomalies (robust z-score vs rolling median)"),
    code("a = detect_anomalies(y); print(len(a), 'flagged days'); a.round(1)"),
    md("## Model comparison (test = last 90 days, never used for choosing the model)"),
    code("m = pd.read_csv('reports/metrics.csv'); m[(m.split=='test')].pivot(index='model', columns='series', values='WAPE').round(3)"),
    code("r = json.load(open('reports/results.json'))['series']\n"
         "for s, v in r.items(): print(s, 'chosen:', v['chosen_model'], '| test:', {k: round(x,3) for k,x in v['test']['chosen'].items()},"
         " '| baseline WAPE:', round(v['test']['baseline']['WAPE'],3))"),
    md("## Failure cases (largest test errors, series `total`)"),
    code("pd.DataFrame(r['total']['worst_test_days']).round(0)"),
]
nb = nbf.v4.new_notebook(cells=cells)
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
nbf.write(nb, "notebooks/demandlab.ipynb")
print("notebook ok")
