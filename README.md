# Leveraging LLMs for Real-World GitHub Issue Resolution

This repository contains the implementation for the project **"Leveraging LLMs for Real-World GitHub Issue Resolution"**, developed as part of CS598 at the University of Illinois Urbana-Champaign. The project explores the potential of Large Language Models (LLMs) in automatically identifying and resolving real-world issues from GitHub repositories using a structured pipeline involving *bug localization* and *patch repair*.

---

## Project Overview

Real-world GitHub issues often require significant human effort to triage and fix. This project presents an automated methodology using multiple LLMs (ChatGPT, Claude, DeepSeek, Grok, Mistral) to:

* **Localize**: Identify files likely containing the bug.
* **Repair**: Generate and apply code patches to resolve the issue.

All experiments are based on the **SWE-Bench Lite** dataset.

---

## Methodology

### 1. Localization Phase

* **Initial File Filtering**: Predict likely file types (.py, .c, .cpp) using issue descriptions.
* **Repository Structure Parsing**: Represent the repo as a tree for improved LLM context.
* **Multi-LLM Predictions**: Query multiple LLMs to independently rank top-K buggy file paths.
* **Union Strategy**: Combine all candidate paths from models to increase coverage.
* **Voting Mechanism**: Use a "Yes/No" agreement step with select models (Claude, GPT, DeepSeek) to refine the final list of buggy files.

### 2. Repair Phase

* **Context Selection**: Choose the granularity of code to patch (file/class/function).
* **Patch Format**: Use a search-and-replace format to improve patch robustness and applicability.

---

## Evaluation

The pipeline is evaluated on SWE-Bench Lite using:

* **Localization Accuracy**: Top-K prediction correctness of buggy files.
* **Patch Precision**: Correctness and relevance of generated code patches.
* **Token Usage**: (Planned) Cost analysis of token consumption per model.

---

## Files & Pipeline Usage

> *Note: Some files may be omitted due to workspace constraints. However, core scripts are provided for full pipeline reproduction.*

### Step-by-Step Instructions

1. **Download Codebases**
   Run `clone_repo.ipynb`
   This notebook will:

   * Clone the necessary GitHub repositories.
   * Checkout the exact commit for each task using SWE-Bench Lite metadata.

2. **Generate Ground Truth**
   Run `generate_groundtruth.ipynb`
   This script creates ground truth labels for buggy file paths in each task.

3. **Set API Keys**

   * Rename `.env.example` to `.env`.
   * Add your API keys for each LLM provider (OpenAI, Anthropic, DeepSeek, etc.).

4. **Filter File Extensions**
   Run `ask_file_extensions.ipynb`
   This notebook uses LLMs to predict the most relevant file extensions based on the issue descriptions.

5. **Bug Localization**
   Run `localize_bug_multi.ipynb`
   This performs multi-LLM bug localization using union and voting strategies to identify likely buggy file paths.

6. **Evaluate Localization**
   Run `evaluate_localization.ipynb`
   Measure the localization accuracy using Top-K metrics against ground truth.

7. **Bug Repair**
   Run `fix_bug.py`
   This script generates and applies bug-fixing patches based on the localization results.

   > ⚠️ **IMPORTANT**: Before running, configure `configs/repair_config.yaml`.
   > You must specify:
   >
   > * LLM model name
   > * Patch output directory
   > * Paths
   > * Whether to use multiprocessing
   >
   > Example configuration is provided in the repo.

8. **Generate Unified Diffs**
   Run `regenerate_diff.py`
   This script reconstructs standard unified diff files based on the patch results:

9. **Final Evaluation & Analysis**
   Run `evaluate.ipynb`
   This notebook summarizes the performance of the results.

---

## Notes on Execution

* Only **DeepSeek-Chat** and **Grok** currently support multiprocessing for batch patching.
* For other models (e.g., GPT, Claude), monitor your API limits (e.g., tokens per minute) before scaling up parallel processes.