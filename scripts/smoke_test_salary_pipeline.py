"""Exercise local LLM extraction through state versioning and certificates."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research_v2.phase2.case_factory import build_clean_case
from research_v2.phase2.commitment_gate import evaluate_commitment
from research_v2.phase2.state_lifecycle import (
    extract_and_ingest_salary_slip,
    mark_risk_reassessed,
)
from research_v2.phase2.workflow_adapter import (
    REQUIRED_CERTIFICATE_TYPES,
    build_workflow_certificates,
)


def main():
    graph_result = {"policy_passed": True, "policy_violations": []}
    case = build_clean_case("salary-pipeline-demo", {"credit_score": 700})
    old_certificates = build_workflow_certificates(case, graph_result)
    text = "Employee: Asha Rao | Employer: Example Pvt Ltd | Net Pay: ₹50,000 | Period: August 2026"

    ingestion = extract_and_ingest_salary_slip(
        case, "uploaded-salary-demo", text, old_certificates
    )
    updated = ingestion.transition.case
    blocked = evaluate_commitment(
        updated, ingestion.transition.certificates,
        REQUIRED_CERTIFICATE_TYPES, True,
    )

    reassessed = mark_risk_reassessed(updated)
    fresh_certificates = build_workflow_certificates(reassessed, graph_result)
    recovered = evaluate_commitment(
        reassessed, fresh_certificates, REQUIRED_CERTIFICATE_TYPES, True
    )
    print(json.dumps({
        "extraction_valid": ingestion.extraction.valid,
        "ingestion_accepted": ingestion.accepted,
        "old_state_version": case.state_version,
        "new_state_version": updated.state_version,
        "old_certificate_statuses": [c.status for c in ingestion.transition.certificates],
        "gate_before_reassessment": blocked.model_dump(),
        "fresh_certificate_types": [c.certificate_type for c in fresh_certificates],
        "gate_after_reassessment": recovered.model_dump(),
    }, indent=2))


if __name__ == "__main__":
    main()
