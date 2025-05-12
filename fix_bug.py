# %%
import os
import re
import json
import datetime
import multiprocessing
import difflib
import logging
from functools import partial
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAIError, RateLimitError, APIError, Timeout
from anthropic import AnthropicError
# Add other provider-specific exceptions as needed
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import OutputParserException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datasets import load_dataset
from pathlib import Path
from collections import defaultdict

# %%
# --- Configuration ---

# Define paths for prompts (adjust as needed)
PROMPT_DIR = Path("./prompts/repair")
# SYSTEM_PROMPT_FILE = PROMPT_DIR / "system_prompt_clean_same.txt"
SYSTEM_PROMPT_FILE = ""
# USER_PROMPT_TEMPLATE_FILE = PROMPT_DIR / "user_prompt_template.txt"
USER_PROMPT_TEMPLATE_FILE = PROMPT_DIR / "full_user_prompt.txt"
# FIX_PATHS_JSON = Path("./ground_truth/bug_paths.json") # Path to your JSON file
# FIX_PATHS_JSON = Path("./localization_candidates/voting_top3union_gptclaudedeepseek_.json") # Path to your JSON file
# FIX_PATHS_JSON = Path("./localization_candidates/union_4voters_r1full.json")
FIX_PATHS_JSON = Path("./localization_candidates/gemini_deepseekr1full_concise.json")
CODEBASE_DIR = Path("./codebases")
EXPERIMENTS_BASE_DIR = Path("./repair_experiments")
CACHE_BASE_DIR = Path("./repair_cache") # <--- ADDED: Base directory for cache

# Hardcoded Configuration
CONFIG = {
    "provider": "anthropic",  # Options: "openai", "anthropic", "deepseek", "google", "mistral", "nvidia", "xai"
    # "provider": "openai",
    # "model_name": "gemini-2.5-flash-preview-04-17",
    # "model_name": "gemini-2.5-pro-exp-03-25",
    # "model_name": "gemini-2.5-pro-preview-03-25",
    # "model_name": "gemini-2.5-pro-preview-05-06",
    "model_name": "claude-3-7-sonnet-latest",
    "temperature": 0,
    "max_tokens": 64000,
    # "max_tokens": 8192,
    "num_processes": 1, # Number of parallel processes
    "system_prompt_path": str(SYSTEM_PROMPT_FILE),
    "user_prompt_template_path": str(USER_PROMPT_TEMPLATE_FILE),
    "fix_paths_json_path": str(FIX_PATHS_JSON),
    "max_retries": 3, # Max retries for API calls
    "use_cache": True, # <--- ADDED: Enable reading from cache
    "update_cache": True, # <--- ADDED: Enable writing to cache
    # "retry_min": 1800, 
    "retry_min": 180, 
    # "retry_max": 3600,
    "retry_max": 360,
}


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
# --- Helper Functions ---

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    logger.info("Environment variables loaded.")

def load_prompt(file_path: Path) -> str:
    """Loads a prompt from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading prompt from {file_path}: {e}")
        raise

def load_json(file_path: Path) -> dict:
    """Loads a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        return {} # Return empty dict if not found, or raise error?
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        raise

def to_valid_file_name(path_str: str) -> str:
    """Converts a path string to a valid filename."""
    # Replace slashes and other problematic characters
    s = re.sub(r'[\\/*?:"<>|]', '_', path_str)
    # Optional: Truncate if too long
    max_len = 100
    if len(s) > max_len:
        s = s[:max_len]
    return s

def get_llm(config: dict):
    """Initializes and returns the Langchain Chat model based on config."""
    provider = config.get("provider", "").lower()
    model_name = config.get("model_name")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 1024) # Note: max_tokens might be interpreted differently by providers

    if not model_name:
        raise ValueError("model_name must be specified in the config")

    logger.info(f"Initializing LLM: Provider={provider}, Model={model_name}, Temp={temperature}, MaxTokens={max_tokens}")

    try:
        if provider == "openai":
            return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
        elif provider == "anthropic":
            # Anthropic uses max_tokens_to_sample, Langchain wrapper might handle this
            return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens) # Check if max_tokens works or need max_tokens_to_sample
        elif provider == "deepseek":
            return ChatDeepSeek(model=model_name, temperature=temperature, max_tokens=max_tokens)
        elif provider == "google":
             # Google uses max_output_tokens
             # REMOVED convert_system_message_to_human=True
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_tokens)
        elif provider == "mistral":
            return ChatMistralAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
        elif provider == "nvidia":
            return ChatNVIDIA(model=model_name, temperature=temperature, max_tokens=max_tokens)
        elif provider == "xai":
            return ChatXAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except ImportError as e:
        logger.error(f"Missing Langchain integration for {provider}. Install required packages. Error: {e}")
        raise
    except Exception as e:
        # Catching potential issues with max_tokens parameter mismatch if Langchain doesn't abstract it fully
        logger.error(f"Error initializing LLM for provider {provider} (check parameter support, e.g., max_tokens vs max_output_tokens): {e}")
        raise


def get_file_content(instance_id: str, relative_path: str) -> str | None:
    """Reads the content of a specific file within the instance's codebase."""
    full_path = CODEBASE_DIR / instance_id / relative_path
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"File not found for {instance_id}: {full_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {full_path}: {e}")
        return None

# --- ADDED: Cache Helper ---
def get_cache_path(config: dict, instance_id: str, file_path: str) -> Path:
    """Constructs the path for a specific cache file."""
    provider = config.get("provider", "unknown_provider")
    model_name = config.get("model_name", "unknown_model")
    valid_model_name = to_valid_file_name(model_name)
    valid_file_path = to_valid_file_name(file_path)
    return CACHE_BASE_DIR / provider / valid_model_name / instance_id / f"{valid_file_path}.txt"

# %%
# Define retryable exceptions for different providers
RETRYABLE_EXCEPTIONS = (
    OpenAIError, # General OpenAI error base class (includes RateLimitError, APIError, Timeout)
    AnthropicError,
    # Add other provider-specific retryable exceptions here
    # Consider adding specific Google, Mistral etc. exceptions if needed for retry
    # TimeoutError, # General timeout
    # ConnectionError, # General connection error
    # Caution: Retrying on broad Exception might retry non-recoverable errors
    Exception # Broad fallback for transient network issues etc. - USE WITH CAUTION
)

@retry(
    stop=stop_after_attempt(CONFIG.get("max_retries", 3)),
    wait=wait_exponential(multiplier=2, min=CONFIG.get("retry_min", 30), max=CONFIG.get("retry_max", 300)), # Example: Exponential backoff from 30min to 1hr
    # wait=wait_exponential(multiplier=2, min=5, max=60), # More reasonable backoff for typical API limits
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    reraise=True # Reraise the exception if all retries fail
)
def invoke_llm_with_retry(llm, messages):
    """Invokes the LLM with retry logic."""
    return llm.invoke(messages)


# --- Parsing Functions (parse_search_replace, parse_search_replace_xml) remain the same ---
# (Keep your existing parse_search_replace_xml function here)
def parse_search_replace_xml(response_content: str) -> list[tuple[str, str]]:
    """
    Parses the LLM response to extract pairs of <original_code_N>
    and <fixed_code_N> blocks using regex. Includes multiple fallback strategies
    for incomplete tags, missing fences, or missing closing tags.
    Returns a list of tuples [(search_block, replace_block), ...].
    """
    pairs = []

    # --- Primary Strategy: Look for complete pairs ---
    primary_pattern = re.compile(
        r"<original_code_(\d+)>([\s\S]*?)</original_code_\1>[\s\n]*"
        r"<fixed_code_\1>([\s\S]*?)</fixed_code_\1>",
        re.MULTILINE
    )
    matches = primary_pattern.findall(response_content)
    if matches:
        pairs = [(match[1].strip(), match[2].strip()) for match in matches]
        logger.info(f"Parsed {len(pairs)} pairs using primary strategy (complete tags).")
        return pairs

    logger.warning("Primary parsing (complete tags) failed. Trying fallback strategies...")

    # --- Fallback Strategy 1: Find first complete <original_code_N>...</original_code_N>, take rest as fixed ---
    fallback_1_pattern = re.compile(
        r"<original_code_(\d+)>([\s\S]*?)</original_code_\1>",
        re.MULTILINE
    )
    first_match_fb1 = fallback_1_pattern.search(response_content)
    if first_match_fb1:
        search_block = first_match_fb1.group(2).strip()
        end_index = first_match_fb1.end()
        potential_replace_block = response_content[end_index:].strip()
        if potential_replace_block.startswith("```"): # Clean potential leading fence
             potential_replace_block = potential_replace_block[3:]
        if potential_replace_block.endswith("```"): # Clean potential trailing fence
            potential_replace_block = potential_replace_block[:-3].strip()

        if search_block and potential_replace_block:
            logger.info("Parsed 1 pair using Fallback Strategy 1 (complete original tag, rest is fixed).")
            return [(search_block, potential_replace_block)]

    # --- Fallback Strategy 3: Handle ONLY opening tags <original_code_N> ... <fixed_code_N> ... ---
    logger.warning("Fallback Strategy 1 failed. Trying Fallback Strategy 3 (opening tags only)...")
    tag_pattern = re.compile(r"<(original_code|fixed_code)_(\d+)>", re.MULTILINE)
    all_tags = [(match.group(1), int(match.group(2)), match.start(), match.end()) for match in tag_pattern.finditer(response_content)]
    all_tags.sort(key=lambda x: x[2]) # Sort by start position

    temp_pairs = []
    i = 0
    while i < len(all_tags) - 1:
        tag_type, tag_num, tag_start, tag_end = all_tags[i]
        next_tag_type, next_tag_num, next_tag_start, next_tag_end = all_tags[i+1]

        if tag_type == "original_code" and next_tag_type == "fixed_code" and tag_num == next_tag_num:
            # Extract raw content first
            raw_search_block = response_content[tag_end:next_tag_start]
            # Find the start of the *next* tag after <fixed_code_N> or end of string
            next_boundary_start = len(response_content)
            if i + 2 < len(all_tags):
                next_boundary_start = all_tags[i+2][2]
            raw_replace_block = response_content[next_tag_end:next_boundary_start]

            # --- ADDED CLEANUP for ``` within extracted blocks ---
            search_block = raw_search_block.strip()
            if search_block.startswith("```"):
                search_block = search_block[3:]
            if search_block.endswith("```"):
                search_block = search_block[:-3]
            search_block = search_block.strip() # Strip again after removing fences

            replace_block = raw_replace_block.strip()
            if replace_block.startswith("```"):
                replace_block = replace_block[3:]
            if replace_block.endswith("```"):
                replace_block = replace_block[:-3]
            replace_block = replace_block.strip() # Strip again
            # --- END ADDED CLEANUP ---

            if search_block or replace_block: # Keep pair even if one part is empty after stripping/cleaning
                 temp_pairs.append((search_block, replace_block))
            i += 2
        else:
            i += 1

    if temp_pairs:
        logger.info(f"Parsed {len(temp_pairs)} pairs using Fallback Strategy 3 (opening tags only, with fence cleanup).")
        return temp_pairs


    # --- Fallback Strategy 2: Remove ``` fences from *entire response* and retry Fallback 1 ---
    logger.warning("Fallback Strategy 3 failed. Trying Fallback Strategy 2 (removing all fences and retry Fallback 1)...")
    cleaned_content = response_content.replace("```", "")
    first_match_fb2 = fallback_1_pattern.search(cleaned_content)
    if first_match_fb2:
        search_block = first_match_fb2.group(2).strip()
        end_index = first_match_fb2.end()
        potential_replace_block = cleaned_content[end_index:].strip()
        # No need to check for ``` here as they were globally removed

        if search_block and potential_replace_block:
            logger.info("Parsed 1 pair using Fallback Strategy 2 (removed all fences, complete original tag).")
            return [(search_block, potential_replace_block)]

    # --- All Strategies Failed ---
    logger.warning("Could not find any parseable pairs using primary or any fallback strategies.")
    return []


# --- Core Repair Logic ---

def repair_file(instance_id: str, path: str, config: dict, llm, system_prompt: str, user_prompt_template: str, problem_statement: str, experiment_path: Path) -> tuple[bool, str | None]:
    """Attempts to repair a single file for a given instance, using cache if enabled."""
    logger.info(f"Attempting to repair {instance_id}: {path}")

    file_content = get_file_content(instance_id, path)
    if file_content is None:
        logger.error(f"Skipping repair for {instance_id}: {path} - Could not read original file.")
        return False, None

    response_content = None # Initialize response content

    # --- Cache Check ---
    cache_path = get_cache_path(config, instance_id, path)
    use_cache = config.get("use_cache", False)

    if use_cache and cache_path.is_file():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                response_content = f.read()
            logger.info(f"Cache hit for {instance_id}:{path}. Using cached response.")
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_path}: {e}. Will proceed to LLM query.")
            response_content = None # Ensure it's None if cache read failed

    # --- LLM Invocation (if cache not hit or not used) ---
    if response_content is None:
        logger.info(f"Cache miss or cache disabled for {instance_id}:{path}. Querying LLM.")
        # --- Prepare Prompts ---
        user_prompt = user_prompt_template.format(
            file_path=path,
            file_content=file_content,
            problem_statement=problem_statement
        )
        # Conditionally add system message
        messages = []
        if system_prompt and system_prompt.strip(): # Check if system_prompt is not empty or just whitespace
            messages.append(SystemMessage(content=system_prompt))
            logger.debug(f"[{instance_id}:{path}] Using system prompt.") # Optional: debug log
        else:
            logger.debug(f"[{instance_id}:{path}] No system prompt provided or it's empty.") # Optional: debug log
        messages.append(HumanMessage(content=user_prompt)) # Always add user prompt

        try:
            # response = invoke_llm_with_retry(llm, messages)
            # response_content = response.content # Store the raw response content

            response_message = invoke_llm_with_retry(llm, messages)


            response_content = ""
            if isinstance(response_message.content, str):
                response_content = response_message.content
            elif isinstance(response_message.content, list):
                print("Info: LLM response content was a list, joining parts.")
                for part in response_message.content:
                    if isinstance(part, str):
                        response_content += (part + "\n")
                    elif isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                        response_content += (part["text"] + "\n")
                    # 嘗試通用的 .text 屬性，這可能涵蓋 TextPart 或類似的物件
                    elif hasattr(part, 'text') and isinstance(getattr(part, 'text'), str):
                        response_content += (getattr(part, 'text') + "\n")
                    else:
                        print(f"Warning: Skipping unexpected part type in list: {type(part)}. Attempting str(part).")
                        try:
                            response_content += (str(part) + "\n") # 備援方案
                        except Exception as e_str:
                            print(f"Could not convert part to string: {e_str}")

                if not response_content and response_message.content:
                    print("Warning: No specific text extraction method worked for list parts, attempting generic join.")
                    try:
                        response_content = "".join(str(p) for p in response_message.content)
                    except Exception as e:
                        print(f"Error joining list parts generically: {e}")
                        response_content = str(response_message.content)
            elif response_message.content is None and hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                response_content = f"LLM initiated tool calls: {response_message.tool_calls}"
                print(response_content)
            else:
                print(f"Warning: LLM response content was of unexpected type: {type(response_message.content)}. Attempting to convert to string.")
                response_content = str(response_message.content if response_message.content is not None else "")



            

            # --- Update Cache (if enabled and successful) ---
            update_cache = config.get("update_cache", False)
            if update_cache and response_content and response_content.strip(): # Only cache non-empty responses
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(response_content)
                    logger.info(f"Saved LLM response to cache: {cache_path}")
                except Exception as e:
                    logger.error(f"Failed to write cache file {cache_path}: {e}")
            elif update_cache:
                logger.warning(f"LLM response for {instance_id}:{path} was empty. Not caching.")


        except Exception as e:
            logger.error(f"LLM invocation failed for {instance_id}:{path} after retries: {e}")
            response_content = f"Error: LLM invocation failed - {e}" # Store error message

            # Save error response to raw_responses (as before)
            raw_response_path = experiment_path / "raw_responses" / instance_id / f"{to_valid_file_name(path)}.error.txt"
            raw_response_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(raw_response_path, "w", encoding="utf-8") as f:
                    f.write(response_content)
            except Exception as e_save:
                logger.error(f"Additionally failed to save error response for {instance_id}:{path}: {e_save}")

            return False, None # Failed LLM invocation

    # --- Ensure response_content is not None before proceeding ---
    if response_content is None:
        logger.error(f"[{instance_id}:{path}] Reached parsing stage with no response_content (Should not happen).")
        return False, None

    # --- Save Raw Response (Always save to experiment dir, regardless of cache) ---
    raw_response_path = experiment_path / "raw_responses" / instance_id / f"{to_valid_file_name(path)}.txt"
    raw_response_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(raw_response_path, "w", encoding="utf-8") as f:
            f.write(response_content)
    except Exception as e:
        logger.error(f"Failed to save raw response for {instance_id}:{path}: {e}")

    # --- Check if response_content indicates an error from LLM failure stage ---
    if response_content.startswith("Error: LLM invocation failed"):
        logger.warning(f"[{instance_id}:{path}] Skipping parsing and diff due to previous LLM invocation error.")
        return False, None

    # --- Parse and Apply Fix ---
    repair_pairs = parse_search_replace_xml(response_content)

    if not repair_pairs:
        logger.warning(f"No repair pairs parsed for {instance_id}:{path}. Raw response saved.")
        # Check if the response was actually empty or just unparseable
        if not response_content.strip():
            logger.warning(f"[{instance_id}:{path}] The response content was empty.")
        # else:
            # logger.warning(f"[{instance_id}:{path}] Response content was not empty but parsing failed.")
        return False, None # Indicate failure as no changes can be applied

    modified_content = file_content # Start with original content
    applied_changes_count = 0
    all_searches_found = True

    # Apply changes sequentially
    for i, (search_block, replace_block) in enumerate(repair_pairs):
        if search_block is None or replace_block is None:
            logger.warning(f"[{instance_id}:{path}] Pair {i+1}: Found None in search/replace block, skipping.")
            continue

        if search_block == replace_block:
            logger.warning(f"[{instance_id}:{path}] Pair {i+1}: Search and replace blocks are identical, skipping.")
            continue # Skip if blocks are the same

        # Check if search_block exists in the *current* state of modified_content
        try:
            # Use find() to check existence without raising error on not found
            if modified_content.find(search_block) != -1:
                # Replace only the *first* occurrence found in the current content state
                modified_content = modified_content.replace(search_block, replace_block, 1)
                applied_changes_count += 1
                logger.info(f"[{instance_id}:{path}] Applied change pair {i+1}")
            else:
                logger.warning(f"Search block for pair {i+1} not found in the current content state for {instance_id}:{path}. May be due to prior replacements or incorrect block from LLM.")
                # Decide if you want to stop or continue with other pairs
                all_searches_found = False
                # break # Option: Stop if any search block isn't found
        except Exception as e:
            logger.error(f"Error applying replacement for pair {i+1} in {instance_id}:{path}: {e}")
            all_searches_found = False
            break # Stop applying further changes if one fails

    if applied_changes_count == 0:
        logger.warning(f"[{instance_id}:{path}] No changes were actually applied (blocks might be identical or not found).")
        # Decide if this is a failure or just no change needed according to LLM
        # Let's consider it a failure for diff generation purposes
        return False, None # No diff to generate if no changes applied

    # --- Generate and Save Diff ---
    diff_path = experiment_path / "diffs" / instance_id / f"{to_valid_file_name(path)}.diff"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_str = ""
    try:
        # Generate diff based on the final modified_content
        diff = difflib.unified_diff(
            file_content.splitlines(keepends=True),
            modified_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
        diff_str = "".join(diff)

        if not diff_str.strip(): # Check if diff is empty or only whitespace
            logger.warning(f"[{instance_id}:{path}] Generated diff is empty, indicating no effective changes were made by the replacements.")
            # Decide if this counts as success or failure
            return False, None # Option: consider empty diff a failure

        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(diff_str)
        logger.info(f"Successfully generated and saved diff ({applied_changes_count} changes applied) for {instance_id}:{path}")
        # Return success even if not all search blocks were found, as long as some changes were applied and a diff was generated?
        return True, diff_str # Return success and the diff string

    except Exception as e:
        logger.error(f"Failed to generate or save diff for {instance_id}:{path}: {e}")
        return False, None # Indicate failure


# --- Multiprocessing Worker ---

def process_instance(instance: dict, config: dict, system_prompt: str, user_prompt_template: str, fix_paths_data: dict, experiment_path: Path) -> dict:
    """Worker function to process a single SWE-bench instance."""
    instance_id = instance['instance_id']
    problem_statement = instance['problem_statement']
    results = {"instance_id": instance_id, "repairs": {}}
    success_count = 0
    total_count = 0

    # Initialize LLM *within* the worker process if needed (some models might have issues with fork)
    # If LLM init is expensive, consider initializing once per process using pool initializer
    try:
        # Pass only necessary config parts if LLM init is sensitive
        llm_config = {
            "provider": config.get("provider"),
            "model_name": config.get("model_name"),
            "temperature": config.get("temperature"),
            "max_tokens": config.get("max_tokens"),
        }
        llm = get_llm(llm_config)
    except Exception as e:
        logger.error(f"[{instance_id}] Failed to initialize LLM in worker: {e}")
        # Return error status for the whole instance
        return {"instance_id": instance_id, "repairs": {}, "error": f"LLM Initialization failed: {e}"}


    paths_to_fix = fix_paths_data.get(instance_id)
    # Ensure paths_to_fix is a list
    if isinstance(paths_to_fix, str):
        paths_to_fix = [paths_to_fix]
    elif not isinstance(paths_to_fix, list): # Handle None or other types
        paths_to_fix = [] # Treat as no paths if not string or list


    if not paths_to_fix:
        logger.warning(f"[{instance_id}] No valid fix paths found in {config['fix_paths_json_path']}. Skipping.")
        results["status"] = "skipped_no_paths"
        return results

    total_count = len(paths_to_fix)
    for path in paths_to_fix:
        # Ensure path is a string before processing
        if not isinstance(path, str) or not path.strip():
            logger.warning(f"[{instance_id}] Skipping invalid path entry: {path}")
            total_count -=1 # Adjust total count if skipping invalid path
            continue

        # Pass the full config to repair_file as it needs cache flags etc.
        success, diff = repair_file(instance_id, path, config, llm, system_prompt, user_prompt_template, problem_statement, experiment_path)
        results["repairs"][path] = {"success": success, "diff_generated": diff is not None and diff.strip() != ""} # Check diff is non-empty
        if success and diff is not None and diff.strip() != "": # Count success only if a non-empty diff was generated
            success_count += 1

    results["status"] = "completed"
    # Handle division by zero if total_count became 0 after skipping invalid paths
    results["success_rate"] = f"{success_count}/{total_count}" if total_count > 0 else "0/0"
    logger.info(f"[{instance_id}] Finished processing. Success rate (non-empty diffs): {results['success_rate']}")
    return results

# %%
# --- generate_combined_diff_json function remains the same ---
# (Keep your existing generate_combined_diff_json function here)
def generate_combined_diff_json(experiment_time_id: str, output_filename: str = "combined_patches.json"):
    """
    Generates a JSON file containing aggregated diff patches for a given experiment run,
    loads the SWE-bench Lite dataset to identify and prints ALL instance_ids
    that did not result in a successful, non-empty diff patch.

    Args:
        experiment_time_id: The timestamp ID of the experiment run.
        output_filename: The name for the output JSON file.
    """
    experiment_path = EXPERIMENTS_BASE_DIR / experiment_time_id
    diffs_dir = experiment_path / "diffs"
    config_path = experiment_path / "config.json"
    output_path = experiment_path / output_filename

    # --- Load expected instance IDs from SWE-bench Lite ---
    all_expected_instance_ids = set()
    try:
        logger.info("Loading SWE-bench Lite dataset to get all instance IDs...")
        # Load the same split used in the main experiment script
        # Added trust_remote_code=True based on potential HF dataset requirements
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", trust_remote_code=True)
        for item in dataset:
            all_expected_instance_ids.add(item['instance_id'])
        logger.info(f"Loaded {len(all_expected_instance_ids)} expected instance IDs from dataset.")
    except Exception as e:
        logger.error(f"Failed to load SWE-bench Lite dataset to get instance IDs: {e}")
        logger.error("Reporting of missing instances might be incomplete.")
        # Continue without the full list? Or raise error? Let's continue but the report will be limited.

    if not diffs_dir.is_dir():
        logger.error(f"Diffs directory not found for experiment '{experiment_time_id}': {diffs_dir}")
        if all_expected_instance_ids:
            logger.info("--- Instances with NO results (missing diff directory) ---")
            for inst_id in sorted(list(all_expected_instance_ids)):
                print(f"  - {inst_id} (Diff directory not found)")
            logger.info("---------------------------------------------------------")
        return # Cannot proceed further

    # --- Try to get model name from config ---
    model_name_or_path = "unknown_model"
    try:
        if config_path.is_file():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                model_name_or_path = config_data.get("model_name", model_name_or_path)
        else:
            logger.warning(f"Config file not found at {config_path}. Using default model name.")
    except Exception as e:
        logger.warning(f"Could not read model name from config file {config_path}: {e}. Using default.")

    # --- Aggregate diffs and track processed instances ---
    aggregated_diffs = defaultdict(str)
    instances_processed_with_dir = set() # Track instances that had a diff directory
    instances_with_empty_diffs = set() # Track instances with dir but no/empty diff files

    instance_dirs = [d for d in diffs_dir.iterdir() if d.is_dir()]

    logger.info(f"Processing {len(instance_dirs)} instance directories found in {diffs_dir}...")
    for instance_dir in instance_dirs:
        instance_id = instance_dir.name
        instances_processed_with_dir.add(instance_id)

        diff_files = sorted(list(instance_dir.glob("*.diff")))

        if not diff_files:
            logger.warning(f"No .diff files found in directory for instance_id: {instance_id}")
            instances_with_empty_diffs.add(instance_id)
            continue

        combined_patch_for_instance = ""
        valid_diff_found = False
        for diff_file in diff_files:
            try:
                with open(diff_file, 'r', encoding='utf-8') as f:
                    diff_content = f.read()
                    # Ensure content is truly non-empty before adding
                    if diff_content and diff_content.strip():
                        # Add newline between concatenated diffs if not already present
                        if combined_patch_for_instance and not combined_patch_for_instance.endswith('\n'):
                             combined_patch_for_instance += '\n'
                        combined_patch_for_instance += diff_content
                        valid_diff_found = True
            except Exception as e:
                logger.error(f"Error reading diff file {diff_file} for instance {instance_id}: {e}")

        if valid_diff_found:
            aggregated_diffs[instance_id] = combined_patch_for_instance
        else:
            logger.warning(f"No non-empty diff content found for instance_id: {instance_id}")
            instances_with_empty_diffs.add(instance_id)

    # --- Identify ALL instances without successful diffs ---
    instances_with_valid_diffs = set(aggregated_diffs.keys())
    all_failed_or_missing_instances = set()

    if all_expected_instance_ids: # Only if dataset was loaded successfully
        # Instances expected but not found in aggregated results
        all_failed_or_missing_instances = all_expected_instance_ids - instances_with_valid_diffs
    else:
        # Fallback: report only those processed but yielded empty diffs
        all_failed_or_missing_instances = instances_processed_with_dir - instances_with_valid_diffs # Those processed but didn't yield valid diff
        logger.warning("Reporting only instances with empty/no diff files found in existing directories, as the full dataset list couldn't be loaded.")


    # --- Print Report ---
    if all_failed_or_missing_instances:
        logger.info(f"--- Instances with NO valid diff generated ({len(all_failed_or_missing_instances)} total) ---")
        # Optionally differentiate reasons
        for inst_id in sorted(list(all_failed_or_missing_instances)):
            reason = ""
            if all_expected_instance_ids: # Check if we have the full list
                if inst_id not in instances_processed_with_dir:
                    reason = "(Diff directory not found or processing failed early)"
                elif inst_id in instances_with_empty_diffs:
                    reason = "(No diff files or only empty diffs found)"
                else:
                    # This case might occur if processing failed before diff generation for an instance dir that exists
                    reason = "(Processing likely failed before generating a diff)"
            elif inst_id in instances_with_empty_diffs: # Fallback if dataset failed loading
                reason = "(No diff files or only empty diffs found)"

            print(f"  - {inst_id} {reason}")
        logger.info("-------------------------------------------------------------------")
    else:
        logger.info("All expected instances seem to have yielded valid diffs.")


    # --- Format output JSON ---
    output_data = []
    # Sort keys for consistent output, ensure instance_id exists in aggregated_diffs
    for instance_id in sorted(list(instances_with_valid_diffs)):
        output_data.append({
            "instance_id": instance_id,
            "model_patch": aggregated_diffs[instance_id],
            "model_name_or_path": model_name_or_path
        })

    if not output_data and not all_failed_or_missing_instances and len(instance_dirs)>0:
        logger.info("Aggregated diff data is empty, but no specific failures were logged for processed instances.")
    elif not output_data:
        logger.warning(f"No data with valid diffs aggregated to write to {output_path}.")
        # Ensure it's an empty list if saving, though it might be better not to save an empty file
        # output_data = []

    # --- Save the combined JSON file (only if there's data) ---
    if output_data:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Successfully generated combined patch file: {output_path} (contains {len(output_data)} instances)")
        except Exception as e:
            logger.error(f"Error writing combined JSON file to {output_path}: {e}")
    elif diffs_dir.is_dir(): # Only log if processing actually happened
        logger.warning(f"No valid diffs were generated in this run. Output file {output_path} will not be created.")


# %%
# --- Main Execution Logic ---

load_environment()

# --- Load Resources ---
logger.info("Loading dataset...")
try:
    # Use streaming=True for large datasets if memory becomes an issue
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", trust_remote_code=True) # Ensure trust_remote_code is set if needed
    # If not streaming, convert to list for multiprocessing, handle potential memory issues
    # dataset_list = list(dataset) # Uncomment and use dataset_list below if needed, watch memory
    dataset_iterable = dataset # Use the iterable directly if possible
    dataset_len = len(dataset) # Get length for tqdm if possible
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    exit(1) # Exit if dataset loading fails


logger.info("Loading prompts and fix paths...")
try:
    # --- Load User Prompt Template (Required) ---
    user_prompt_template = load_prompt(Path(CONFIG["user_prompt_template_path"]))

    # --- Load System Prompt (Optional) ---
    system_prompt_path_str = CONFIG.get("system_prompt_path", "").strip() # Get path, default to empty string
    system_prompt = "" # Default to empty string
    if system_prompt_path_str:
        system_prompt_path = Path(system_prompt_path_str)
        if system_prompt_path.is_file():
            try:
                loaded_prompt = load_prompt(system_prompt_path)
                if loaded_prompt and loaded_prompt.strip(): # Check if loaded prompt is not empty/whitespace
                    system_prompt = loaded_prompt
                    logger.info(f"Loaded system prompt from: {system_prompt_path}")
                else:
                    logger.warning(f"System prompt file found but is empty: {system_prompt_path}. Using empty system prompt.")
            except Exception as e_load:
                logger.warning(f"Failed to load system prompt from {system_prompt_path}: {e_load}. Using empty system prompt.")
        else:
            logger.warning(f"System prompt file path specified but not found: {system_prompt_path}. Using empty system prompt.")
    else:
        logger.info("No system prompt path specified. Proceeding without a system prompt.")


    # --- Load Fix Paths Data ---
    fix_paths_data = load_json(Path(CONFIG["fix_paths_json_path"]))
except Exception as e:
    logger.error(f"Failed to load prompts or fix paths: {e}")
    exit(1)


if not fix_paths_data:
    logger.warning(f"Fix paths file ({CONFIG['fix_paths_json_path']}) is empty or invalid. Processing might yield no results.")
    # Decide whether to continue or exit (continuing for now)

# --- Setup Experiment Directory ---
time_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_path = EXPERIMENTS_BASE_DIR / time_id
raw_response_dir = experiment_path / "raw_responses"
diff_dir = experiment_path / "diffs"

try:
    experiment_path.mkdir(parents=True, exist_ok=True)
    raw_response_dir.mkdir(exist_ok=True)
    diff_dir.mkdir(exist_ok=True)
    logger.info(f"Experiment directory created: {experiment_path}")
except Exception as e:
    logger.error(f"Failed to create experiment directories: {e}")
    exit(1)

# --- Save Config ---
config_save_path = experiment_path / "config.json"
try:
    with open(config_save_path, 'w', encoding='utf-8') as f:
        # Convert Path objects in config to strings for JSON serialization
        config_to_save = {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()}
        json.dump(config_to_save, f, indent=4)
    logger.info(f"Experiment config saved to {config_save_path}")
except Exception as e:
    logger.error(f"Failed to save config: {e}")
    # Continue execution even if config saving fails? (Potentially problematic for reproducibility)


# --- Run Multiprocessing ---
num_processes = CONFIG.get("num_processes", 1)
logger.info(f"Starting repair process with {num_processes} workers...")

# Create a partial function with fixed arguments for the worker
worker_func = partial(
    process_instance,
    # Pass the full config dict
    config=CONFIG,
    system_prompt=system_prompt,
    user_prompt_template=user_prompt_template,
    fix_paths_data=fix_paths_data,
    experiment_path=experiment_path
)

all_results = []

# Use multiprocessing pool
try:
    # Use imap_unordered for potentially better performance as results come in
    # Wrap the dataset_iterable (or dataset_list) with the pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm to show progress over the dataset items
        all_results = list(tqdm(pool.imap_unordered(worker_func, dataset_iterable), total=dataset_len, desc="Processing Instances"))
except Exception as e:
    logger.error(f"Multiprocessing pool encountered an error: {e}")
    # Decide how to handle partial results if pool fails mid-way


# --- Save Aggregate Results (Optional but recommended) ---
results_summary_path = experiment_path / "results_summary.json"
try:
    with open(results_summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Aggregate results summary saved to {results_summary_path}")
except Exception as e:
    logger.error(f"Failed to save aggregate results summary: {e}")


logger.info("Repair process finished.")

# --- Generate Combined Diff JSON ---
logger.info("Generating combined diff file...")
try:
    generate_combined_diff_json(time_id)
except Exception as e:
    logger.error(f"Failed during combined diff generation: {e}")

logger.info("Script finished.")