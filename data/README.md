# Data Directory

Place `diabetic_data.csv` in this folder before training.

## Download Dataset

**UCI ML Repository — Diabetes 130-US Hospitals (1999–2008)**

🔗 https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Or via the `ucimlrepo` package:
```python
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=296)
df = dataset.data.original
df.to_csv('data/diabetic_data.csv', index=False)
```

## File Format

- 101,766 rows × 50 columns
- Target column: `readmitted` (values: `NO`, `>30`, `<30`)
