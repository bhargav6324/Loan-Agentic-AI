"""Typed evidence structures used by the Phase 2 experiments.

These schemas deliberately sit outside the production LangGraph state. They
provide a stable, auditable representation for synthetic document cases and
later LLM extraction results.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


DocumentType = Literal[
    "loan_application",
    "salary_slip",
    "bank_statement",
    "employment_letter",
    "identity_record",
]
EvidenceScalar = Union[str, int, float, bool]


class EvidenceField(BaseModel):
    """One extracted value together with its provenance."""

    model_config = ConfigDict(extra="forbid")

    field_name: str = Field(min_length=1)
    value: EvidenceScalar
    source_document: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class DocumentEvidence(BaseModel):
    """A document and the fields extracted from it."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    document_type: DocumentType
    present: bool = True
    fields: Dict[str, EvidenceField] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """Workflow facts needed to test structural faults."""

    model_config = ConfigDict(extra="forbid")

    policy_node_executed: bool = False
    risk_node_executed: bool = False
    risk_score_state_version: Optional[int] = Field(default=None, ge=0)
    critic_node_executed: bool = False
    commitment_attempted: bool = False


class LoanEvidenceCase(BaseModel):
    """A reproducible synthetic case used by Phase 2 experiments."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    state_version: int = Field(default=1, ge=1)
    applicant_data: Dict[str, EvidenceScalar] = Field(default_factory=dict)
    documents: List[DocumentEvidence] = Field(default_factory=list)
    extracted_evidence: Dict[str, EvidenceField] = Field(default_factory=dict)
    execution: ExecutionTrace = Field(default_factory=ExecutionTrace)
    fault_labels: List[str] = Field(default_factory=list)


class EvidenceValidationResult(BaseModel):
    """Deterministic validation output; it never makes a loan decision."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    normalized_evidence: Dict[str, EvidenceField] = Field(default_factory=dict)
