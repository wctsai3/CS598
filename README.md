# Leveraging LLMs for Real-World GitHub Issue Resolution

This repository contains the implementation for the project **"Leveraging LLMs for Real-World GitHub Issue Resolution"**, developed as part of CS598 at the University of Illinois Urbana-Champaign. The project explores the potential of Large Language Models (LLMs) in automatically identifying and resolving real-world issues from GitHub repositories using a structured pipeline involving *bug localization* and *patch repair*.

## Project Overview

Real-world GitHub issues often require significant human effort to triage and fix. This project presents an automated methodology using multiple LLMs (ChatGPT, Claude, DeepSeek, Grok, Mistral) to:

- **Localize**: Identify files likely containing the bug.
- **Repair**: Generate and apply code patches to resolve the issue.

All experiments are based on the **SWE-Bench Lite** dataset.

## Methodology

### 1. Localization Phase

- **Initial File Filtering**: Predict likely file types (.py, .c, .cpp) using issue descriptions.
- **Repository Structure Parsing**: Represent repo as a tree for better LLM context.
- **Multi-LLM Predictions**: Query multiple LLMs to independently rank top-K buggy file paths.
- **Union Strategy**: Combine all candidate paths from models to increase coverage.
- **Voting Mechanism**: Refine results with a "Yes/No" voting step from select models (Claude, GPT, DeepSeek) to keep the most agreed-upon paths.

### 2. Repair Phase

- **Context Selection**: Determine relevant code chunk size (file/class/function).
- **Patch Strategy**: Use a search-and-replace format instead of diff to improve accuracy and reliability in code patching.

## Evaluation

Experiments were conducted on SWE-Bench Lite and evaluated using:
- **Localization Accuracy** (Top-K file prediction success)
- **Patch Precision** (correctness and placement of generated patches)
- **Token Usage** (planned analysis for resource cost)

## Files Explanation

> *Note: Some files may be missing from this repository due to workspace constraints. The uploaded code includes the core clean scripts. Please follow the steps below to reproduce the full pipeline.*

### Step-by-Step Usage

1. **Download Codebases**  
   Run `clone_repo.ipynb`  
   This notebook will:
   - Clone the relevant GitHub repositories for each task.
   - Checkout to the correct commit based on the SWE-Bench Lite dataset.

2. **Generate Ground Truth**  
   Run `generate_groundtruth.ipynb`  
   This script creates the ground truth labels for buggy file paths per task.

3. **Set API Keys**  
   - Rename `.env.example` to `.env`
   - Insert your API keys for all the LLM providers you're using (e.g., OpenAI, Anthropic, DeepSeek, etc.).

4. **Filter File Extensions**  
   Run `ask_file_extensions.ipynb`  
   This notebook prompts LLMs to determine relevant file extensions based on the issue description and narrows down the search space.

5. **Bug Localization**  
   Run `localize_bug_multi.ipynb`  
   This core notebook performs multi-model localization using union and voting strategies to identify likely buggy file paths.

6. **Evaluate Localization**  
   Run `evaluate_localization.ipynb`  
   Evaluate the localization accuracy (Top-K metrics) against ground truth using the model outputs.

7. **Bug Repair**  
   Run `repair_multi.ipynb`  
   Attempt to repair the identified bugs by generating and applying patches.

### Notes on Execution

- Only **DeepSeek-Chat** and **Grok** support multiprocessing for batch processing.
- For other models (e.g., GPT, Claude), consider your API tier and **tokens-per-minute (TPM)** limits before adjusting the number of concurrent processes.