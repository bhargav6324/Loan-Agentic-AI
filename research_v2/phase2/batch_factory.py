"""Convert existing structured applicants into Phase 2 evidence cases."""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from .case_factory import build_clean_case
from .fault_catalog import FaultType
from .fault_injector import inject_fault
from .schemas import LoanEvidenceCase


def build_cases_from_dataframe(
    dataframe: pd.DataFrame,
    limit: Optional[int] = None,
    seed: int = 42,
) -> List[LoanEvidenceCase]:
    """Build deterministic clean cases without exposing target labels to evidence."""

    frame = dataframe
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        frame = dataframe.sample(n=min(limit, len(dataframe)), random_state=seed)
    frame = frame.sort_values("id")

    cases = []
    excluded = {"target_default"}
    for row in frame.to_dict(orient="records"):
        applicant_id = str(row["id"])
        applicant_data = {key: value for key, value in row.items() if key not in excluded}
        cases.append(build_clean_case(f"applicant-{applicant_id}", applicant_data))
    return cases


def build_paired_cases(
    clean_cases: Iterable[LoanEvidenceCase],
) -> List[LoanEvidenceCase]:
    """Return one clean case plus one identical-seed variant per fault."""

    paired = []
    for clean in clean_cases:
        paired.append(clean)
        paired.extend(inject_fault(clean, fault) for fault in FaultType)
    return paired

