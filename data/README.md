# Clinical Consultation QA Dataset Documentation

## 1. Dataset Overview

This dataset was created to train and evaluate sequence-to-sequence transformer models (specifically Google FLAN-T5) on the task of extracting structured clinical summaries from multi-turn doctor-patient consultation dialogues.

In clinical practice, doctor-patient conversations are recorded or transcribed as unstructured text. To integrate with Electronic Health Record (EHR) systems, medical staff require specific clinical categories to be extracted concisely and accurately.

## 2. Dataset Variants & Directory Layout

The `data/` directory is organized as follows:

```
data/
├── README.md               # This documentation file
├── raw/
│   ├── dataset_finetune_s.csv      # Full raw dataset (1,500 QA pairs across 150 dialogue contexts)
│   ├── 500_medical_qa_dataset.csv  # 500-sample milestone snapshot
│   └── dataset_finetune_s(old).csv # Initial prototype snapshot
├── processed/
│   ├── train.csv           # 1,050 samples (sample-level random split; 100% dialogue overlap)
│   ├── val.csv             # 225 samples
│   └── test.csv            # 225 samples
└── grouped/
    ├── train_grouped.csv   # 1,050 samples (105 dialogues, zero context leakage)
    ├── val_grouped.csv     # 220 samples (22 dialogues, zero context leakage)
    └── test_grouped.csv    # 230 samples (23 dialogues, zero context leakage)
```

## 3. Data Schema & Fields

### Raw Schema (`data/raw/dataset_finetune_s.csv`)

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `Question` | `string` | Query targeting a specific clinical category | `List the patient's chief complaints with comma separated value.` |
| `Context` | `string` | Full multi-turn transcript of a doctor-patient consultation | `Good afternoon! What brings you in today?...` |
| `Answer` | `string` | Ground truth extracted medical entity/summary | `Hair loss, Stress, Thyroid issues.` |

### Processed Schema (`data/processed/*.csv`)

For sequence-to-sequence training (FLAN-T5), question and context are concatenated into standard prompt format:

| Column Name | Type | Description | Format |
| :--- | :--- | :--- | :--- |
| `input` | `string` | Formatted seq2seq model input | `Question: {Question} Context: {Context}` |
| `target` | `string` | Expected target generation | `{Answer}` |

## 4. Clinical Target Categories

The questions systematically extract 10 standardized clinical sections:
1. **Brief History**: Summary of onset, progression, and relevant timeline.
2. **Chief Complaints**: Primary reason(s) for the visit formatted as comma-separated values.
3. **Past Medical History**: Chronic conditions, prior non-surgical diagnoses.
4. **Hospitalization & Surgical History**: Previous hospital admissions and surgeries.
5. **Gynecological / Obstetric History**: OB/GYN background where applicable (or "No data available").
6. **Lifestyle & Social Activity**: Activity level, occupational sedentary time, diet, exercise, smoking/alcohol habits.
7. **Family History**: Health conditions and genetic predispositions in immediate relatives.
8. **Allergies & Hypersensitivity**: Drug allergies (e.g. penicillin, sulfa) or food allergies.
9. **Previous Investigations**: Prior blood tests, imaging, or diagnostics.
10. **Review of Systems (ROS)**: Systemic symptoms across cardiovascular, respiratory, gastrointestinal, or neurological domains.

## 5. Synthetic vs. Real Data Disclosures

- **Data Nature**: Demonstration / Synthetic Clinical Dialogues.
- **Privacy & Provenance Disclosures**: These conversations are synthetic doctor-patient dialogues designed to emulate clinical encounters without containing real patient health records or real clinical identifiers. No formal legal guarantee of statutory HIPAA or GDPR compliance certification is made or implied.
- **Generation Method**: Synthesized dialogue interactions paired with expert-annotated clinical category answers (see `src/data/generator.py`).

## 6. Train / Validation / Test Splitting Methodology

- **Split Ratio**: 70% Train (1,050 samples), 15% Validation (225 samples), 15% Test (225 samples).
- **Exact Match Overlap**: 0 samples (zero identical input-target pairs between splits).
- **Context-Level Observation**: Splitting was performed at the question-answer level. Because each dialogue context is queried across multiple clinical categories, individual dialogues appear with different questions across the splits. For benchmark evaluation on completely unseen patient profiles, grouped dialogue splitting is also supported via `src/data/dataset.py`.

