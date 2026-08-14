"""Exact durable schemas and constants for Symile-MIMIC data artifacts."""

from __future__ import annotations

import pyarrow as pa

DATASET_ID = "symile"
DATASET_RELEASE = "1.0.0"
TASK_ID = "pneumonia_strict"
LABEL_POLICY_VERSION = "symile-pneumonia-strict-v1"
OFFICIAL_SPLITS = ("train", "validation", "test")
DEVELOPMENT_SPLITS = ("train", "validation")
REPEAT_SEEDS = (17, 42, 2026)
OUTER_FOLDS = tuple(range(5))
LAB_ITEM_IDS = (
    "50802",
    "50804",
    "50813",
    "50818",
    "50820",
    "50821",
    "50861",
    "50862",
    "50863",
    "50868",
    "50878",
    "50882",
    "50885",
    "50893",
    "50902",
    "50910",
    "50912",
    "50931",
    "50934",
    "50947",
    "50960",
    "50970",
    "50971",
    "50983",
    "51006",
    "51133",
    "51146",
    "51200",
    "51221",
    "51222",
    "51237",
    "51244",
    "51248",
    "51249",
    "51250",
    "51254",
    "51256",
    "51265",
    "51274",
    "51275",
    "51277",
    "51279",
    "51301",
    "51678",
    "52069",
    "52073",
    "52074",
    "52075",
    "52135",
    "52172",
)
LAB_NAMES = {
    "50802": "Base Excess",
    "50804": "Calculated Total CO2",
    "50813": "Lactate",
    "50818": "pCO2",
    "50820": "pH",
    "50821": "pO2",
    "50861": "Alanine Aminotransferase (ALT)",
    "50862": "Albumin",
    "50863": "Alkaline Phosphatase",
    "50868": "Anion Gap",
    "50878": "Asparate Aminotransferase (AST)",
    "50882": "Bicarbonate",
    "50885": "Bilirubin, Total",
    "50893": "Calcium, Total",
    "50902": "Chloride",
    "50910": "Creatine Kinase (CK)",
    "50912": "Creatinine",
    "50931": "Glucose",
    "50934": "H",
    "50947": "I",
    "50960": "Magnesium",
    "50970": "Phosphate",
    "50971": "Potassium",
    "50983": "Sodium",
    "51006": "Urea Nitrogen",
    "51133": "Absolute Lymphocyte Count",
    "51146": "Basophils",
    "51200": "Eosinophils",
    "51221": "Hematocrit",
    "51222": "Hemoglobin",
    "51237": "INR(PT)",
    "51244": "Lymphocytes",
    "51248": "MCH",
    "51249": "MCHC",
    "51250": "MCV",
    "51254": "Monocytes",
    "51256": "Neutrophils",
    "51265": "Platelet Count",
    "51274": "PT",
    "51275": "PTT",
    "51277": "RDW",
    "51279": "Red Blood Cells",
    "51301": "White Blood Cells",
    "51678": "L",
    "52069": "Absolute Basophil Count",
    "52073": "Absolute Eosinophil Count",
    "52074": "Absolute Monocyte Count",
    "52075": "Absolute Neutrophil Count",
    "52135": "Immature Granulocytes",
    "52172": "RDW-SD",
}

SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("subject_id", pa.int64(), nullable=False),
        pa.field("hadm_id", pa.int64(), nullable=False),
        pa.field("official_split", pa.string(), nullable=False),
        pa.field("source_row", pa.int64(), nullable=False),
        pa.field("pneumonia_state", pa.int8(), nullable=True),
        pa.field("age_years", pa.int16(), nullable=False),
        pa.field("sex", pa.string(), nullable=False),
        pa.field("view_position", pa.string(), nullable=False),
    ]
)

LAB_SCHEMA = pa.schema(
    [pa.field("sample_id", pa.string(), nullable=False)]
    + [pa.field(f"lab_{item_id}_value", pa.float64(), nullable=True) for item_id in LAB_ITEM_IDS]
    + [pa.field(f"lab_{item_id}_observed", pa.bool_(), nullable=False) for item_id in LAB_ITEM_IDS]
)

CV_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("repeat_seed", pa.int32(), nullable=False),
        pa.field("outer_fold", pa.int8(), nullable=False),
    ]
)


def task_contract() -> dict[str, object]:
    """Return the frozen strict-pneumonia source-state interpretation."""
    return {
        "task_id": TASK_ID,
        "label_source": "symile_mimic_data.csv:Pneumonia",
        "label_policy_version": LABEL_POLICY_VERSION,
        "positive": "pneumonia_state == 1",
        "negative": "pneumonia_state == 0",
        "excluded": ["pneumonia_state == -1", "pneumonia_state is null"],
    }
