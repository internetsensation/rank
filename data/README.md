# Data directory

This directory is intentionally ignored by git except for placeholder files.

Expected price schema after cleaning:

```text
date,ticker,open,high,low,close,adj_close,volume
```

Recommended canonical format inside the package:

- pandas DataFrame
- MultiIndex: `date`, `ticker`
- Columns: `open`, `high`, `low`, `close`, `adj_close`, `volume`
