# Gemini Project Analysis: PestCLEF 2026

## Project Overview
**PestCLEF 2026** is a BioNLP shared task focused on document-level relation extraction from the EPOP (European and Mediterranean Plant Protection Organization) document collection. The goal is to extract structured knowledge graphs representing relationships between pests, diseases, plants, locations, and other entities.

The project is developed by **Mihai Radulescu (Team BioNLP412)** and is hosted as a Kaggle competition.

## Architectural Summary
The codebase is highly modular and designed for iterative improvement:

1.  **Core Data Layer (`src/pestclef/data.py`, `schema.py`):**
    - Standardizes the heterogeneous JSON annotations into a unified `Document` schema.
    - Implements robust canonical entity construction and alias alignment.
    - Handles text splitting and span normalization.

2.  **Baseline Models (`src/pestclef/model.py`):**
    - **NumPy Baseline:** A zero-dependency, pure NumPy multi-label linear classifier. This ensures the project is runnable in any Python environment, even those with broken library stacks.
    - **Sklearn Baseline:** A slightly more advanced baseline using logistic regression.

3.  **ModernBERT Integration (`src/pestclef/modernbert.py`):**
    - Implements a staged transformer pipeline.
    - **Mention Detection:** A token classification model (ModernBERT-base) with Focal Loss to handle class imbalance.
    - **Relation Extraction:** A multi-label sequence classification model that uses entity-aware markers (e.g., `<SUBJ_Pest>`, `<OBJ_Plant>`) to provide context to the encoder.

4.  **Orchestration & Evaluation (`src/pestclef/pipeline.py`, `evaluation.py`):**
    - Provides a unified interface for training, cross-validation, and submission generation.
    - Produces detailed error reports, including false positive/negative analysis and mention detection failures.

## Strong Points
- **Excellent Modularity:** The clean separation between the data layer and the model layer allows for seamless swapping of different encoder architectures (e.g., replacing the NumPy classifier with a Transformer).
- **Environment Resilience:** The inclusion of a pure NumPy baseline is a sophisticated fallback strategy for restricted compute environments (like Kaggle kernels or specific HPC nodes).
- **Comprehensive Evaluation:** The pipeline doesn't just output a score; it generates deep insights into *where* the model is failing (mention recall, entity alignment, or relation logic).
- **Schema-Aware Reasoning:** The implementation of `RelationSchema` ensures that the model only predicts biologically plausible relations (e.g., preventing a "Vector" from being "Located in" a "Disease").

## Weak Points
- **Mention Detection Bottleneck:** Initial baseline results (Micro F1 ~0.15) suggest that the dictionary-based mention detection is the primary bottleneck for end-to-end performance.
- **Low Support for Minority Classes:** Sparse relations like *Transmits* or *Causes* likely suffer from extreme class imbalance, which even Focal Loss may struggle to fully resolve without data augmentation.
- **Context Window Limits:** The current relation context uses a fixed sentence radius, which might miss long-range dependencies in complex EPOP documents.

## Potential Future Steps
1.  **Hybrid Mention Detection:** Fully transition to the neural mention detector while using the dictionary-based lexicon as a high-precision candidate generator (Hybrid Mode).
2.  **Long-Context Encoders:** Explore ModernBERT's native support for longer sequences or implement hierarchical encoding to capture document-level context more effectively.
3.  **Data Augmentation:** Implement back-translation or LLM-based synthetic data generation to bolster the training set for rare relation types.
4.  **Coreference Resolution:** Improve the "coref guess" logic with a dedicated neural coreference module to better aggregate entity mentions.
5.  **Ensemble Cascade:** Combine the ModernBERT predictions with a reasoning layer (e.g., a Graph Neural Network or an LLM-based verifier) to filter borderline cases.

## Analysis of Best Run (v9) & Improvement Potential

The current best run (**modernbert_e2e_v9**) achieved a Kaggle score of **0.43679**. Detailed analysis of the `dev` artifacts (Micro F1: 0.215) reveals the following:

### 1. The Mention Detection Bottleneck
The `dev_candidate_recall.json` shows that **61.2% (401/655)** of missed relations are due to **"bad_span"** (mention detection failure).
- **Date (F1: 0.23)** and **Dissemination_pathway (F1: 0.20)** are the weakest entity types.
- **Location (Recall: 0.48)** is missing over half of the ground-truth locations, which severely impacts the most common relation, *Located_in*.
- **Vector (F1: 0.07)** is almost entirely missed, explaining the **0.0 F1** on the *Transmits* relation.

### 2. Candidate Generation & Pruning
- **Candidate Recall is low (0.296).** Even if the relation classifier was perfect, it could only achieve ~30% recall given the current entities.
- **Pruning Losses:** 40 valid relations (6.1% of total) were lost because the entity pair was pruned before reaching the relation model (likely due to distance or type constraints).

### 3. Relation Classification Insights
- **Precision vs. Recall:** The model has low precision across the board (Micro Precision: 0.27). For *Located_in*, there are **154 False Positives** vs. only **48 True Positives**.
- **Clustering Noise:** 20 failures were attributed to "bad_clustering_or_canonicalization," where unrelated mentions were merged or noise was promoted to a canonical entity.
- **Calibration:** The threshold for most relations is set to **0.48**, which might be too aggressive given the high false-positive rate.

### 4. V10: Config & Heuristics Tuning Results

The **v10** run (Config & Heuristics Tuning) yielded a Micro F1 of **0.196**. Key observations compared to v9:

- **Candidate Recall Improvement:** Relaxing the pruning distances (`max_sentence_distance: 5`, `max_layout_distance: 10`) successfully reduced pruning misses from **40 (6.1%)** to **28 (4.3%)**.
- **Lexicon Precision Trade-off:** Increasing `mention_hybrid_lexicon_confidence` to **0.65** improved recall for major relations like *Located_in* (from 0.12 to 0.17) and *Causes* (from 0.33 to 0.40). However, this came at the cost of significantly lower precision (Micro Precision fell from 0.27 to 0.20), resulting in a slight overall F1 drop.
- **Mention Quality:** Neural mention detection precision for *Pests* and *Plants* decreased, suggesting that the expanded distance limits might be introducing more noise into the relation classifier's context windows.
- **Denylist Effectiveness:** The expanded denylists helped stabilize the `Location` F1 (0.57 vs 0.58 in v9), despite the increased noise from relaxed pruning.

### 5. V11: Precision Recovery & Context Depth Results

The **v11** evaluation (Context Radius 2, Reverted Distances, Rule-based NER) yielded a Micro F1 of **0.200**.

- **Precision Recovery:** Reverting to v9 distance settings successfully brought Micro Precision back to **0.26** (up from v10's 0.20).
- **Context Impact:** The increased `relation_context_sentence_radius: 2` significantly improved the **Affects** relation (F1: **0.25** vs v9's 0.15), suggesting the model benefits from the wider 5-sentence window.
- **Rule-based NER Failure:** The specialized Date/Vector regex injection **decreased performance**. Date F1 dropped to **0.10** (from v9's 0.23) and total "bad spans" increased to 425. This indicates the regex patterns were too noisy and likely suppressed better neural predictions during span merging.
- **Support Gap:** *Transmits* and *Dispersed_by* remains at 0.0 F1, confirming that context and distance alone cannot fix the lack of training examples for these relations.

### 6. V12: Context Retention & NER Refinement Results

The **v12** evaluation (Context Radius 2, Removed Rule NER, Relation-Specific Thresholds) yielded a Micro F1 of **0.184**.

- **NER Stability:** Removing the noisy rule-based extractors successfully restored **Date F1** to **0.22** (from v11's 0.10), confirming the neural model is better than simple regex for this task.
- **Threshold Precision Gain:** Increasing the threshold for *Located_in* to **0.55** improved its precision slightly, but significantly reduced recall, leading to a net loss in F1 for this major relation.
- **Plateau:** The continued 0.0 F1 for *Transmits* and *Dispersed_by* suggests that "tuning" has reached its limit. The model simply lacks enough positive training signal for these rare classes to overcome the calibration margins.

### 7. V13: Surgical Data Augmentation & Oversampling Results

The **v13** evaluation (Context Radius 2, Oversampling Ratio 0.1, Contextual Jittering) yielded a Micro F1 of **0.163**.

- **Minority Relation "Wake-up":** The primary goal was achieved—minority relations started predicting. **Dispersed_by** achieved a non-zero F1 (**0.08**), and **Causes** maintained high recall (**0.44**).
- **Precision Collapse (Hallucination):** The model began hallucinating minority relations aggressively. **Transmits** generated **75 False Positives** (vs 0 in v9) while still missing all 5 True Positives. **Affects** and **Causes** also saw FP counts more than double compared to v9.
- **Aggressive Biasing:** Oversampling combined with jittering likely made the model over-sensitive to specific entity type pairs (e.g., Pest + Vector), causing it to predict *Transmits* even when the sentence context didn't support it.
- **NER Consistency:** The neural mention detection remains stable (F1 ~0.55), confirming that the relation extraction layer is the source of the recent fluctuations.

### 8. V14: Hard Negative Mining Results

The **v14** evaluation (Oversampling Ratio 0.05, Hard Negative Ratio 1.0) yielded a Micro F1 of **0.226**, making it the **best-performing local configuration** so far (surpassing v9's 0.215).

- **Hallucination Suppressed:** The targeted hard negative mining successfully anchored the model's predictions. False Positives for *Transmits* dropped from **75** (in v13) back down to **4**, proving that the model learned to rely on textual context rather than just entity type combinations.
- **Precision Improvement:** Overall Micro Precision recovered strongly to **0.269**, pulling the F1 score up.
- **Relation Gains:** The expanded context window from previous iterations, combined with balanced sampling, led to major improvements in the **Affects** relation (F1 jumped to **0.295** vs v9's 0.155) and **Located_in** (F1 **0.204** vs v9's 0.165).
- **Residual Weaknesses:** *Transmits* and *Dispersed_by* still struggle with recall, returning 0 True Positives, but the model is no longer wildly guessing them.

### 9. Summary of Scores & Next Steps

| Version | Description | Dev Micro F1 | Kaggle Score |
| :--- | :--- | :--- | :--- |
| v9 | Best Neural Baseline | 0.215 | 0.43679 |
| v10 | Config Tuning (Relaxed) | 0.196 | 0.36185 |
| v11 | Precision & Context | 0.200 | N/A |
| v12 | Threshold Refinement | 0.184 | N/A |
| v13 | Augmentation & Oversampling| 0.163 | N/A |
| v14 | Hard Negative Mining | **0.226** | TBD |

### Actionable Next Steps

1.  **Generate Kaggle Submission:** The v14 model demonstrates a clear, structurally sound improvement over v9 (the previous best). Generate a submission using `--train-on-dev` to capture its performance on the public leaderboard.
2.  **Ensemble Consideration:** If v14's Kaggle score is lower than v9 due to hidden test-set distribution differences, an ensemble blending v9's high precision on major classes with v14's improved *Affects* and *Located_in* reasoning could be the optimal path forward.

---
*Updated by Gemini CLI on 2026-04-20.*
