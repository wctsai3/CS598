import os
import json
from openai import OpenAI
from .utils import *
from .prompts import *

def prepare_bug_localize_batch_requests(dataset, config):
    """Prepare all tasks for batch processing."""
    batch_tasks = []
    
    for task in dataset:
        instance_id = task["instance_id"]
        problem_statement = task["problem_statement"]
        patch = task["patch"]
        codebase_path = f"/home/tweichuan/project/codebases/{instance_id}"

        # Create prompt for the LLM
        tree_string = generate_tree_string(
            codebase_path,
            show_descriptions=config["show_descriptions"],
            file_extensions=config["file_extensions"],
            ignored_dirs=config["ignored_dirs"]
        )
        prompt = create_buggy_file_localization_prompt(instance_id, tree_string, problem_statement, codebase_path)
        
        batch_tasks.append({
            "custom_id": instance_id,
            "messages": [{"role": "user", "content": prompt}], 
            "config": {
                "model": config['model_configs'][config['llm']]["model"], 
                "temperature": config['model_configs'][config['llm']]["temperature"],
                "max_tokens": config['model_configs'][config['llm']]["max_tokens"]
            }
        })
    
    return batch_tasks

def prepare_openai_batch_file(batch_tasks, output_dir):
    """Prepare a JSONL file for OpenAI's batch API."""
    # Create batch directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine batch file path
    batch_file_path = os.path.join(output_dir, "openai_batch.jsonl")
    
    # Write the batch file
    with open(batch_file_path, 'w', encoding='utf-8') as f:
        for batch_task in batch_tasks:
            # Create entry for each task

            entry = {
                "custom_id": batch_task["custom_id"],
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {
                    "model": batch_task["config"]["model"],
                    "messages": batch_task["messages"],
                    "temperature": batch_task["config"]["temperature"],
                    "max_tokens": batch_task["config"]["max_tokens"]
                }
            }
            f.write(json.dumps(entry) + '\n')
    
    print(f"OpenAI batch file created: {batch_file_path}")
    return batch_file_path

def submit_openai_batch(batch_file_path):
    """Submit a batch job to OpenAI's batch API and return the batch ID."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        with open(batch_file_path, 'rb') as f:
            file = client.files.create(
                file=f,
                purpose="batch"
            )

        batch = client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        batch_id = batch.id
        print(f"OpenAI batch submitted successfully. Batch ID: {batch_id}")
        return batch_id

    except Exception as e:
        print(f"Error submitting OpenAI batch: {str(e)}")
        return None

def check_openai_batch_status(batch_id):
    """Check the status of an OpenAI batch job."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        batch = client.batches.retrieve(batch_id)
        return batch.status
    except Exception as e:
        print(f"Error checking OpenAI batch status: {str(e)}")
        return "error"


def download_openai_batch_results(batch_id, output_dir):
    """Download results from a completed OpenAI batch job."""
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Download the output
        output_file = f"{output_dir}/{batch_id}.jsonl"
        client.batches.outputs.retrieve(
            batch_id=batch_id,
            file_path=output_file
        )
        
        # Parse the results
        responses = {}
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                instance_id = result.get("input_id")
                if instance_id and "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    # Save individual response
                    response_path = os.path.join(output_dir, f"{instance_id}.txt")
                    with open(response_path, "w", encoding="utf-8") as rf:
                        rf.write(content)
                    
                    responses[instance_id] = content
        
        print(f"Downloaded {len(responses)} results from OpenAI batch")
        return responses
    
    except Exception as e:
        print(f"Error downloading OpenAI batch results: {str(e)}")
        return {}

def prepare_and_submit_batch(batch_tasks, llm_name, output_dir):
    """Prepare batch file for the specified LLM and submit it."""
    if llm_name == "chatgpt":
        batch_file_path = prepare_openai_batch_file(batch_tasks, output_dir)
        batch_id = submit_openai_batch(batch_file_path)
    
    else:
        raise ValueError(f"Unsupported LLM: {llm_name}")
    
    # Save batch info
    batch_info = {
        "llm": llm_name,
        "batch_file_path": batch_file_path,
        "batch_id": batch_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "submitted" if batch_id else "prepared"
    }
    
    batch_info_path = os.path.join(output_dir, "batch_info.json")
    with open(batch_info_path, "w", encoding="utf-8") as f:
        json.dump(batch_info, f, indent=2)
    
    return batch_id, batch_file_path

def download_batch_results(batch_id, llm_name, output_dir):
    """Download batch processing results based on LLM type."""
    if llm_name == "chatgpt":
        return download_openai_batch_results(batch_id, output_dir)
    else:
        print(f"No batch results download available for {llm_name}")
        return {}