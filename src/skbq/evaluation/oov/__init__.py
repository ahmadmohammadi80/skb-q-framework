"""Out-of-vocabulary evaluation framework for SKB-Q."""

from skbq.evaluation.oov.analysis import (
    candidate_recall,
    operator_distance_statistics,
    oov_coverage,
    split_reproducibility_report,
)
from skbq.evaluation.oov.oov_dataset import OOVDataset, OOVEvaluationRecord
from skbq.evaluation.oov.protocols import (
    GNNBaselineHook,
    OOVProtocolResult,
    OracleComparisonProtocol,
    RandomFallbackHook,
    SeenOperatorReconstructionProtocol,
    StructuralNearestNeighborHook,
    UnseenOperatorBridgingProtocol,
)
from skbq.evaluation.oov.split import (
    VocabularySplit,
    architecture_family_holdout,
    operator_class_holdout,
)

__all__ = [
    "GNNBaselineHook",
    "OOVDataset",
    "OOVEvaluationRecord",
    "OOVProtocolResult",
    "OracleComparisonProtocol",
    "RandomFallbackHook",
    "SeenOperatorReconstructionProtocol",
    "StructuralNearestNeighborHook",
    "UnseenOperatorBridgingProtocol",
    "VocabularySplit",
    "architecture_family_holdout",
    "candidate_recall",
    "operator_class_holdout",
    "operator_distance_statistics",
    "oov_coverage",
    "split_reproducibility_report",
]
