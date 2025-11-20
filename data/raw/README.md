# Raw Clinical Consultation QA Datasets

This directory contains the original raw tabular datasets used for training and evaluating sequence-to-sequence models (FLAN-T5) on clinical information extraction from doctor-patient consultation transcripts.

---

## 1. File Inventory & Description

| File Name | Samples | Dialogues | Schema | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **`dataset_finetune_s.csv`** | **1,500** | **150** | `[Question, Context, Answer]` | **Primary Dataset**: The full benchmark corpus used to generate both `data/processed/` (sample-level split) and `data/grouped/` (dialogue-isolated split). Contains 150 unique consultation transcripts evaluated across 10 distinct clinical queries. |
| **`500_medical_qa_dataset.csv`** | 500 | 50 | `[Question, Context, Answer]` | **Milestone Dataset**: An intermediate 500-sample snapshot (50 dialogues $\times$ 10 questions) used during early baseline fine-tuning experiments and pipeline validation. |
| **`dataset_finetune_s(old).csv`** | 500 | 50 | `[Question, Context, Answer]` | **Prototype Snapshot**: An initial experimental prototype preserved for research traceability and historical checkpoint replication. |

---

## 2. Data Schema & Specifications

All raw CSV files share a standardized three-column schema with **zero missing (null) values**:

| Column Name | Data Type | Description | Representative Example |
| :--- | :--- | :--- | :--- |
| **`Question`** | `string` | The targeted clinical query prompting extraction of a specific category. | `"List the patient's chief complaints with comma separated value."` |
| **`Context`** | `string` | Full multi-turn transcript of the doctor-patient dialogue. | `"Good afternoon! How are you feeling today? I've been experiencing lower back pain for about three weeks..."` |
| **`Answer`** | `string` | Expert-curated ground truth summary or comma-separated entity list. | `"Lower back pain, sharpness, tingling sensation."` |

---

## 3. The 10 Standardized Clinical Target Sections

Each consultation dialogue in `dataset_finetune_s.csv` is queried across 10 standardized Electronic Health Record (EHR) categories (150 samples per category):

1. **Chief Complaints**: Primary symptoms formatted as comma-separated values (`List the patient's chief complaints with comma separated value.`).
2. **Current Symptoms & Timeline**: Narrative onset and symptom progression (`Summarize the patient's current symptoms and their medical background, including past diagnoses.`).
3. **Current and Past Health Issues**: Broad medical background overview (`Summarize the patient's current and past health issues.`).
4. **Past Medical History**: Chronic illnesses and pre-existing diagnoses (`Summarize the patient's past medical history.`).
5. **Hospitalization & Surgical History**: Inpatient admissions, surgeries, and procedures (`Summarize the patient's hospitalization and surgical history.`).
6. **Gynecological History**: Obstetric and gynecological history where applicable, or standard absent declaration (`Extract and summarize the patient's gynecological history from the context.`).
7. **Lifestyle & Social Activity**: Occupational sedentary time, exercise habits, diet, smoking, and alcohol consumption (`Summarize the patient's lifestyle and social activity.`).
8. **Family History**: Hereditary conditions and family medical risk factors (`Summarize the patient's family history.`).
9. **Allergies & Hypersensitivities**: Drug allergies (e.g., penicillin, sulfa) and food reactions (`Summarize the patient's have any known allergies, hypersensitivities, or adverse reactions to medications?`).
10. **Previous Investigations**: Prior diagnostics, laboratory tests, or imaging results (`Summarize the patient's previous investigations of patient.`).

---

## 4. Preprocessing & Downstream Splits

The raw datasets are transformed by `src/data/dataset.py` into model-ready sequence-to-sequence format (`Question: {Question} Context: {Context}` $\rightarrow$ `{Answer}`):

* **Sample-Level Split (`data/processed/`)**:
  * `train.csv`: 1,050 samples (70%)
  * `val.csv`: 225 samples (15%)
  * `test.csv`: 225 samples (15%)
  * *Characteristics*: 0 identical QA pairs between splits; in-domain section routing benchmark.
* **Grouped Dialogue Split (`data/grouped/`)**:
  * `train_grouped.csv`: 1,050 samples (105 dialogues, 70%)
  * `val_grouped.csv`: 220 samples (22 dialogues, ~15%)
  * `test_grouped.csv`: 230 samples (23 dialogues, ~15%)
  * *Characteristics*: Strict dialogue context isolation (0% context overlap) for assessing generalizability to completely unseen patient dialogues.

---

## 5. Synthetic Nature & Privacy Disclosures

* **Synthetic Personas**: Dialogue transcripts feature synthetic patient profiles (*Selena Smith*, *Daniel Thompson*, *Sarah Johnson*, etc.) designed to simulate clinical encounters without using Protected Health Information (PHI).
* **Identifier Checks**: Automated scans for Social Security Numbers, phone numbers, email addresses, and date-of-birth patterns returned zero matches across all rows.
* **Intended Use**: This corpus is structured for natural language processing research, sequence-to-sequence model fine-tuning, and clinical NLP benchmarks.

