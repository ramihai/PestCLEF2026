# Implementation plan for PestCLEF2026 shared task

## High level phases
Phase 1: Data Parsing & Serialization
Phase 2: Hybrid Graph Encoding
Phase 3: Relational Segmentation
Phase 4: Optimization & Augmentation
Phase 5: Ensemble & LLM Cascade
Phase 6: Kaggle Submission Pipeline

## Detailed steps

### Phase 1: Data Parsing, JSON Ingestion, and Entity Alignment

#### Step 1.1: JSON Ingestion and Text Mapping
Goal: Map the structured JSON annotations to their corresponding raw text documents.
Inputs: PestCLEF-2026_train.json, PestCLEF-2026_dev.json, PestCLEF-2026_test.json - all in the data/json directory - and the raw text documents in the data/EPOP_documents/ directory (separated by dev/test/train).
Outputs: A unified data structure linking doc_id to its raw text string and corresponding JSON graph arrays.
Constraints: Must handle potential encoding issues from translated French-to-English text.
Files to create/modify: src/data/dataset_loader.py
Definition of Done: The data loader successfully iterates over all document IDs and returns a paired dictionary of {doc_id: {"text": "...", "relations": [...]}}.

#### Step 1.2: Gold-Standard Graph and Structure Parsing
Goal: Extract predefined relations directly from the JSON structures to formulate target training variables, and parse text into token, sentence, and paragraph structures.
Inputs: Unified data structure from Step 1.1.
Outputs: Indexed structural blocks (tokens mapped to sentences) and explicit ground-truth node/edge relationships.
Constraints: Exact surface forms from the JSON subject and object keys must be used as the ground-truth nodes to bypass manual taxonomy linking during training.
Files to create/modify: src/data/parser.py, src/data/tokenizer.py
Definition of Done: Text is successfully tokenized using a BioNLP-adapted tokenizer (e.g., SciLitLLM or BioBERT), with sentences bounded and mapped directly to the JSON predicate targets.

#### Step 1.3: EDA
Goal: Perform exploratory data analysis on the ingested data. Calculate statistic information and generate tables, graphs and visualizations. Look into class imbalance, group by relation types (binary/3-ary/etc) and relation classes ("located in", "affects" etc).
Inputs: Data output by Step 1.2.
Outputs: tables, graphs and visualizations
Constraints: Data must be already present, no augmentation.
Files to create/modify: src/EDA.ipynb
Definition of Done: Relevant information is extracted from the data and presented in a useful format for exploratory data analysis.

### Phase 2: Hybrid Graph-Transformer Architecture

#### Step 2.1: Contextual Representation Encoding
Goal: Extract deep contextualized embeddings for all tokens and aggregate them into unified representations for entity nodes.
Inputs: Tokenized documents from Phase 1.
Outputs: High-dimensional embedding vectors for tokens and pooled entities.
Constraints: Max sequence lengths must be managed carefully (e.g., chunking or using long-context models) to prevent memory overflows.
Files to create/modify: src/models/transformer_encoder.py
Definition of Done: A forward pass outputs a dense representation tensor for all tokens, and an attention-weighted pooling function successfully groups coreferenced entity spans into singular vectors.

#### Step 2.2: Hierarchical Tree Graph (HTG) Instantiation
Goal: Construct a multi-level graph architecture connecting Document, Sentence, and Entity nodes.
Inputs: Pooled entity embeddings, sentence-level token embeddings.
Outputs: Graph Attention Network (GAT) updated node embeddings.
Constraints: Bidirectional edges must connect Entity nodes to their specific Sentence nodes, and all Sentence nodes must connect to the global Document node.
Files to create/modify: src/models/htg_layer.py
Definition of Done: Graph convolution layers execute message passing across the document structure without dimension mismatch errors, outputting contextually enriched entity node embeddings.

### Phase 3: Relational Segmentation and Hypergraph Reasoning

#### Step 3.1: Binary Table-Filling and Criss-Cross Attention
Goal: Capture inter-pair dependencies and transitive logic across the document for binary relations.
Inputs: GAT-updated entity embeddings.
Outputs: An N*N feature matrix of relation probabilities.
Constraints: Must utilize a Criss-Cross Attention module to dynamically cross-reference row and column features within the tensor.
Files to create/modify: src/models/segmentation_module.py
Definition of Done: The module calculates interaction probabilities for every possible entity pair in the document simultaneously.

#### Step 3.2: Differentiable Rule Injection and N-Ary Extraction
Goal: Apply biological schema constraints to zero-out impossible relations and predict complex n-ary events.
Inputs: The N*N relation matrix and localized entity embeddings.
Outputs: A biologically constrained probability matrix and bipartite event node predictions.
Constraints: A hard mask must be applied to the loss function preventing structurally impossible relations (e.g., enforcing that a Vector cannot 'Affect' a Location).
Files to create/modify: src/models/rules_mask.py, src/models/n_ary_hypergraph.py
Definition of Done: Network predictions strictly conform to the 5 predefined EPOP relation schemas, and multi-argument events are successfully classified via localized MLPs.

### Phase 4: Optimization & Augmentation

#### Step 4.1: Adaptive Focal Loss and Hard Negative Sampling
Goal: Prevent majority-class collapse caused by the extreme frequency of the Located in relation compared to sparse relations like Transmits.
Inputs: Raw logits from Phase 3 and ground-truth JSON targets.
Outputs: Calculated loss gradients.
Constraints: Must use Focal Loss configured with inverse frequency weights derived from the training set. Random negative sampling must be replaced by selecting syntactically close but biologically unlinked entity pairs.
Files to create/modify: src/training/loss_functions.py, src/training/trainer.py
Definition of Done: The training loop converges smoothly, and the validation metrics show non-zero recall for minority classes (e.g., Causes, Transmits).

### Phase 5: Post-Processing, Ensembling, and LLM Cascade

#### Step 5.1: Triplet Voting and LLM Verification
Goal: Maximize precision on the blind test set using model consensus and zero-shot LLM verification for borderline cases.
Inputs: Individual model predictions; raw document text for edge cases.
Outputs: A finalized, high-confidence set of extracted triplets.
Constraints: Only retain triplets that meet a strict voting threshold across the ensemble.
Files to create/modify: src/inference/ensemble_voter.py, src/inference/llm_verifier.py
Definition of Done: The pipeline automatically detects low-confidence ensemble predictions, constructs a prompt with surrounding sentences, and queries an LLM API with majority-semantic voting to finalize the relation.

### Phase 6: Kaggle Submission Pipeline Compliance

#### Step 6.1: JSON Serialization and DataFrame Compilation
Goal: Package the finalized knowledge graph relations into the exact CSV format mandated by Kaggle.
Inputs: Finalized triplets for all documents in PestCLEF-2026_test.json.
Outputs: A submission.csv file.
Constraints: The output DataFrame must contain exactly two columns: doc_id and knowledge_graph. The knowledge_graph column must be a strictly serialized JSON string of the extracted triplet objects.
Files to create/modify: scripts/generate_submission.py
Definition of Done: Running the script produces a submission.csv that uses json.dumps() for the graph column and correctly uses index=False during CSV generation , passing Kaggle's local format checks.
