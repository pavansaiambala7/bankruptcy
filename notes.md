# Development Notes

## ARFF Format
- Lines starting with % are comments
- @attribute lines give column names
- @data marks the start of rows
- Missing values encoded as ?
- class column arrives as bytes: b'0' or b'1'

## Class Parsing
df["class"] = df["class"].astype(str).str.extract(r"(\d+)")[0].astype(int)

## Class Imbalance
Only ~5% of companies go bankrupt.
Handle with scale_pos_weight (XGBoost) and class_weight (Keras).

## Fix
Use str.extract(r'(\d+)') to get the digit from b'0' or '0'.
