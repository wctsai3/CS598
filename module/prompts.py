

def create_buggy_file_localization_prompt(instance_id, problem_statement, tree_string, codebase_path):
    """Create prompt for asking LLM to localize buggy files."""
    
    # Generate tree representation

    # tree_string = generate_tree_string(
    #     codebase_path,
    #     show_descriptions=config["show_descriptions"],
    #     file_extensions=config["file_extensions"],
    #     ignored_dirs=config["ignored_dirs"]
    # )
    
    # Create the prompt with focus on ranked path listing
    prompt = f"""
# Bug Localization Task for {instance_id}

## Problem Statement
{problem_statement}

## Codebase Structure
```
{tree_string}
```

## Task
Based on the problem statement and codebase structure, identify the Python files that are most likely to contain the bug related to the problem statement.

IMPORTANT: List the file paths in descending order of likelihood (most likely first). Do not include explanations, just provide the ranked list of file paths.

Format your response as follows:
1. [top 1 file path]
2. [top 2 file path]
3. [top 3 file path]
...
10. [top 10 file path]

Your response should include ONLY Python files (.py) with their full paths relative to the root of the codebase.
"""

    return prompt