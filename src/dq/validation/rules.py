import pandas as pd


def rule_positive(df: pd.DataFrame, col: str, greater_than: float = 0):
    """Values must be > greater_than."""
    return df[col] <= greater_than, f"{col}_positive", f"{col} must be greater than {greater_than}", col


def rule_non_null(df: pd.DataFrame, col: str):
    """Values must not be null/empty."""
    mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
    return mask, f"{col}_non_null", f"{col} must not be null or empty", col


def rule_not_future_date(df: pd.DataFrame, col: str):
    """Dates must not be in the future."""
    now = pd.Timestamp.now(tz=None)
    mask = df[col] > now
    return mask, f"{col}_not_future", f"{col} must not be in the future", col


def rule_pattern(df: pd.DataFrame, col: str, pattern: str, desc: str):
    """Values must match regex pattern."""
    mask = ~df[col].astype(str).str.fullmatch(pattern)
    return mask, f"{col}_pattern", desc, col


def rule_length(df: pd.DataFrame, col: str, min_len: int | None = None, max_len: int | None = None):
    """Values must satisfy length bounds."""
    lengths = df[col].astype(str).str.len()
    mask = False
    if min_len is not None:
        mask |= lengths < min_len
    if max_len is not None:
        mask |= lengths > max_len
    return mask, f"{col}_length", f"{col} length must be between {min_len} and {max_len}", col


def rule_duplicates(df: pd.DataFrame, subset: list[str], name: str | None = None):
    """
    Flag rows that have duplicate values on the given subset of columns.
    """
    mask = df.duplicated(subset=subset, keep=False)
    col_label = ",".join(subset)
    rule_name = name or f"duplicate_on_{col_label}"
    desc = f"Duplicate values detected on columns [{col_label}]"
    return mask, rule_name, desc, col_label
