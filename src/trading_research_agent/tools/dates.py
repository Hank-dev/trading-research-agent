from datetime import date, timedelta


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def split_date_range(
    start_date: str, end_date: str, lockbox_pct: float
) -> tuple[str, str]:
    """Split a date range into a train/validation segment and a held-out lockbox.

    Returns (train_end_date, lockbox_start_date) where the lockbox is the trailing
    `lockbox_pct` fraction of the total span. The split day belongs to the lockbox
    (i.e. train ends the day before lockbox starts) so the two segments do not overlap.
    """
    if not 0.0 < lockbox_pct < 1.0:
        raise ValueError("lockbox_pct must be between 0 and 1 (exclusive)")

    start = parse_iso_date(start_date)
    end = parse_iso_date(end_date)
    total_days = (end - start).days
    if total_days < 2:
        raise ValueError("Date range is too short to split")

    lockbox_days = max(1, int(round(total_days * lockbox_pct)))
    if lockbox_days >= total_days:
        raise ValueError("lockbox_pct leaves no training data")

    lockbox_start = end - timedelta(days=lockbox_days)
    train_end = lockbox_start - timedelta(days=1)
    return train_end.isoformat(), lockbox_start.isoformat()
