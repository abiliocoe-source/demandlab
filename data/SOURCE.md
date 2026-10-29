# Data source
- **Dataset:** Bike Sharing Dataset (daily file `day.csv`, 731 days, 2011-01-01..2012-12-31), UCI Machine Learning Repository, id 275.
- **URL:** https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset (downloaded from the zip link in `demandlab/data.py`).
- **Origin:** Capital Bikeshare system, Washington D.C., merged with weather and holiday data. Authors: Hadi Fanaee-T and Joao Gama.
- **License / citation:** the bundled `raw/UCI_Readme.txt` requires citing Fanaee-T & Gama (2013), "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence, doi:10.1007/s13748-013-0040-3. The UCI page lists the dataset under CC BY 4.0 (check the page before redistributing).
- **Limitations:** one city and two years (only two seasonal cycles); daily aggregates; no demand censoring information (sold-out/unavailable bikes look like low demand); 2012 growth reflects system expansion, not just demand; a single storm (late Oct 2012) falls in the test period.
- **Fallback:** if `day.csv` is unavailable, `demandlab.data.synthetic_demand()` produces FICTIONAL data (used in tests only).
