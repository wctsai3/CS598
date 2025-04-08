import os
import re
import ast
import json
import time
import asyncio
import datetime
from pathlib import Path

def setup_batch_directories(llm_name):
    """Create necessary directories for storing batch requests and results."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    time_id = f"{timestamp}"
    
    base_dir = Path(f"/home/tweichuan/project/batch_requests/{llm_name}/{time_id}")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    return str(base_dir), time_id

def extract_modified_file_path(patch):
    """Extracts the modified file path from the first line of a Git diff."""
    match = re.search(r'diff --git a/(.*?) b/', patch)
    return match.group(1) if match else None

def extract_module_description(file_path):
    """Extract the module-level docstring from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for triple-quoted docstrings at the module level
        docstring_pattern = re.compile(r'^(?:(?:#[^\n]*\n)*)?(?:\"\"\"|\'\'\')(.*?)(?:\"\"\"|\'\'\')(?:\s*$|\s*\n)', 
                                      re.DOTALL | re.MULTILINE)
        
        match = docstring_pattern.search(content)
        if match:
            # Extract the docstring and clean it up
            docstring = match.group(1).strip()
            
            # Convert multiline docstring to a single line
            docstring = '   '.join([line.strip() for line in docstring.split('\n')])
            return docstring
    except Exception:
        pass
    
    return None

def get_folder_description(path, show_descriptions=False):
    """Get the description for a folder from its __init__.py file."""
    if not show_descriptions:
        return None
        
    init_path = os.path.join(path, "__init__.py")
    if os.path.isfile(init_path):
        return extract_module_description(init_path)
    
    return None

# def get_tree(path, indent="", is_last=True, file_extensions=None, ignored_dirs=None, show_descriptions=False):
#     """Generate a text-based tree representation of directory structure."""
#     name = os.path.basename(path)
    
#     # Skip ignored directories
#     if ignored_dirs and os.path.isdir(path) and name.lower() in ignored_dirs:
#         return []
    
#     # Check if it's a file or directory
#     if os.path.isdir(path):
#         # Get all items in the directory
#         try:
#             items = os.listdir(path)
#         except PermissionError:
#             return []
            
#         items.sort()
        
#         # Check if this directory contains any Python files
#         contains_python = False
#         for item in items:
#             item_path = os.path.join(path, item)
#             if os.path.isfile(item_path) and os.path.splitext(item)[1].lower() in file_extensions:
#                 contains_python = True
#                 break
#             elif os.path.isdir(item_path) and not (ignored_dirs and item.lower() in ignored_dirs):
#                 # Check subdirectories recursively
#                 subtree = get_tree(item_path, "", True, file_extensions, ignored_dirs, show_descriptions)
#                 if subtree:
#                     contains_python = True
#                     break
        
#         # If no Python files in this directory or subdirectories, skip it
#         if not contains_python:
#             return []
        
#         # Create the line for the current directory
#         line = indent
#         line += "└── " if is_last else "├── "
#         line += name
        
#         # Add description if available
#         description = get_folder_description(path, show_descriptions)
#         if description:
#             line += f"  # {description}"
        
#         result = [line]
        
#         # Process each item
#         valid_items = []
#         for item in items:
#             item_path = os.path.join(path, item)
            
#             # Skip hidden files starting with "."
#             if item.startswith('.'):
#                 continue
                
#             # Skip ignored directories
#             if ignored_dirs and os.path.isdir(item_path) and item.lower() in ignored_dirs:
#                 continue
                
#             # Only include python files
#             if os.path.isfile(item_path):
#                 ext = os.path.splitext(item)[1].lower()
#                 if ext in file_extensions:
#                     valid_items.append(item)
#             else:
#                 # Include directories that might contain python files
#                 subtree = get_tree(item_path, "", True, file_extensions, ignored_dirs, show_descriptions)
#                 if subtree:
#                     valid_items.append(item)
        
#         # Process valid items
#         for i, item in enumerate(valid_items):
#             item_path = os.path.join(path, item)
            
#             # Determine if this is the last item
#             is_last_item = (i == len(valid_items) - 1)
            
#             # Create new indent for the next level
#             new_indent = indent + ("    " if is_last else "│   ")
            
#             # Add the item to the result
#             subtree = get_tree(item_path, new_indent, is_last_item, file_extensions, ignored_dirs, show_descriptions)
#             result.extend(subtree)
        
#         return result
#     else:
#         # If it's a file, just return the line
#         ext = os.path.splitext(name)[1].lower()
#         if ext in file_extensions:
#             line = indent
#             line += "└── " if is_last else "├── "
#             line += name
#             return [line]
#         else:
#             return []

def get_tree(path, indent="", is_last=True, file_extensions=None, ignored_dirs=None, show_descriptions=False, is_root=True, hide_root=True):
    """Generate a text-based tree representation of directory structure."""
    name = os.path.basename(path)

    if ignored_dirs and os.path.isdir(path) and name.lower() in ignored_dirs:
        return []

    if os.path.isdir(path):
        try:
            items = os.listdir(path)
        except PermissionError:
            return []

        items.sort()
        contains_python = False
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path) and ((os.path.splitext(item)[1].lower() in file_extensions) or (not file_extensions)):
                contains_python = True
                break
            elif os.path.isdir(item_path) and not (ignored_dirs and item.lower() in ignored_dirs):
                subtree = get_tree(item_path, "", True, file_extensions, ignored_dirs, show_descriptions, False)
                if subtree:
                    contains_python = True
                    break

        if not contains_python:
            return []

        result = []
        if not is_root or not hide_root:
            line = indent
            line += "└── " if is_last else "├── "
            line += name
            description = get_folder_description(path, show_descriptions)
            if description:
                line += f"  # {description}"
            result = [line]
        else:
            result = [] if hide_root else [name]

        valid_items = []
        for item in items:
            item_path = os.path.join(path, item)
            if item.startswith('.'):
                continue
            if ignored_dirs and os.path.isdir(item_path) and item.lower() in ignored_dirs:
                continue
            if os.path.isfile(item_path):
                ext = os.path.splitext(item)[1].lower()
                if (ext in file_extensions) or (not file_extensions):
                    valid_items.append(item)
            else:
                subtree = get_tree(item_path, "", True, file_extensions, ignored_dirs, show_descriptions, False)
                if subtree:
                    valid_items.append(item)

        for i, item in enumerate(valid_items):
            item_path = os.path.join(path, item)
            is_last_item = (i == len(valid_items) - 1)
            
            if is_root and hide_root:
                new_indent = ""
            else:
                new_indent = indent + ("    " if not is_root and is_last else "│   ")

            subtree = get_tree(item_path, new_indent, is_last_item, file_extensions, ignored_dirs, show_descriptions, False)
            result.extend(subtree)

        return result
    else:
        ext = os.path.splitext(name)[1].lower()
        if (ext in file_extensions) or (not file_extensions):
            line = indent
            line += "└── " if is_last else "├── "
            line += name
            return [line]
        else:
            return []

# def get_tree(path, indent="", is_last=True, file_extensions=None, ignored_dirs=None, show_descriptions=False, is_root=True):
#     """Generate a text-based tree representation of directory structure."""
#     name = os.path.basename(path)
    
#     if ignored_dirs and os.path.isdir(path) and name.lower() in ignored_dirs:
#         return []

#     if os.path.isdir(path):
#         try:
#             items = os.listdir(path)
#         except PermissionError:
#             return []
        
#         items.sort()
#         contains_python = False
#         for item in items:
#             item_path = os.path.join(path, item)
#             if os.path.isfile(item_path) and os.path.splitext(item)[1].lower() in file_extensions:
#                 contains_python = True
#                 break
#             elif os.path.isdir(item_path) and not (ignored_dirs and item.lower() in ignored_dirs):
#                 subtree = get_tree(item_path, "", True, file_extensions, ignored_dirs, show_descriptions, False)
#                 if subtree:
#                     contains_python = True
#                     break
        
#         if not contains_python:
#             return []

#         # ✅ Only add the └── line if it's NOT the root
#         result = []
#         if not is_root:
#             line = indent
#             line += "└── " if is_last else "├── "
#             line += name
#             description = get_folder_description(path, show_descriptions)
#             if description:
#                 line += f"  # {description}"
#             result = [line]
#         else:
#             result = [name]  # Just print the folder name (without └──)

#         valid_items = []
#         for item in items:
#             item_path = os.path.join(path, item)
#             if item.startswith('.'):
#                 continue
#             if ignored_dirs and os.path.isdir(item_path) and item.lower() in ignored_dirs:
#                 continue
#             if os.path.isfile(item_path):
#                 ext = os.path.splitext(item)[1].lower()
#                 if ext in file_extensions:
#                     valid_items.append(item)
#             else:
#                 subtree = get_tree(item_path, "", True, file_extensions, ignored_dirs, show_descriptions, False)
#                 if subtree:
#                     valid_items.append(item)

#         for i, item in enumerate(valid_items):
#             item_path = os.path.join(path, item)
#             is_last_item = (i == len(valid_items) - 1)
#             new_indent = indent + ("    " if not is_root and is_last else "│   ")
#             subtree = get_tree(item_path, new_indent, is_last_item, file_extensions, ignored_dirs, show_descriptions, False)
#             result.extend(subtree)

#         return result
#     else:
#         ext = os.path.splitext(name)[1].lower()
#         if ext in file_extensions:
#             line = indent
#             line += "└── " if is_last else "├── "
#             line += name
#             return [line]
#         else:
#             return []

def generate_tree_string(path, show_descriptions=False, file_extensions=None, ignored_dirs=None):
    """Generate a string representation of the directory tree."""
    tree_lines = get_tree(
        path, 
        file_extensions=file_extensions, 
        ignored_dirs=ignored_dirs, 
        show_descriptions=show_descriptions
    )
    
    if tree_lines:
        return "\n".join(tree_lines)
    else:
        return "No Python files found."

def build_tree_dict(path, file_extensions=None, ignored_dirs=None, show_descriptions=False):
    """Build a JSON tree representation of directory structure."""
    name = os.path.basename(path)
    
    # Skip ignored directories
    if ignored_dirs and os.path.isdir(path) and name.lower() in ignored_dirs:
        return None
    
    if os.path.isdir(path):
        # Get all items in the directory
        try:
            items = os.listdir(path)
        except PermissionError:
            return None
            
        items.sort()
        
        # Check if this directory contains any Python files
        contains_python = False
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path) and os.path.splitext(item)[1].lower() in file_extensions:
                contains_python = True
                break
            elif os.path.isdir(item_path) and not (ignored_dirs and item.lower() in ignored_dirs):
                # Check subdirectories recursively
                child = build_tree_dict(item_path, file_extensions, ignored_dirs, show_descriptions)
                if child:
                    contains_python = True
                    break
        
        # If no Python files in this directory or subdirectories, skip it
        if not contains_python:
            return None
        
        result = {"name": name, "type": "directory", "children": []}
        
        # Add description if available
        if show_descriptions:
            description = get_folder_description(path, show_descriptions)
            result["description"] = description if description else ""
        
        for item in items:
            item_path = os.path.join(path, item)
            
            # Skip hidden files
            if item.startswith('.'):
                continue
                
            # Skip ignored directories
            if ignored_dirs and os.path.isdir(item_path) and item.lower() in ignored_dirs:
                continue
                
            # Process files and directories
            child = build_tree_dict(item_path, file_extensions, ignored_dirs, show_descriptions)
            if child:
                result["children"].append(child)
        
        return result
    else:
        # If it's a file, check if it's a Python file
        ext = os.path.splitext(name)[1].lower()
        if ext in file_extensions:
            return {"name": name, "type": "file"}
        else:
            return None

def generate_tree_json(path, show_descriptions=False, file_extensions=None, ignored_dirs=None):
    """Generate a JSON representation of the directory tree."""
    tree_dict = build_tree_dict(
        path, 
        file_extensions=file_extensions, 
        ignored_dirs=ignored_dirs, 
        show_descriptions=show_descriptions
    )
    
    if tree_dict:
        return json.dumps(tree_dict, indent=2)
    else:
        return json.dumps({"error": "No Python files found."})

def get_bug_paths():
    with open('/home/tweichuan/project/ground_truth/bug_paths.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_skeleton(code: str) -> str:
    tree = ast.parse(code)
    skeleton_lines = []

    def indent(level):
        return "    " * level

    # ✅ Import 處理（支援相對匯入）
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                line = f"import {alias.name}" if not alias.asname else f"import {alias.name} as {alias.asname}"
                skeleton_lines.append(line)
        elif isinstance(node, ast.ImportFrom):
            dot_prefix = "." * node.level
            module = node.module or ""
            full_module = f"{dot_prefix}{module}"
            names = ", ".join([
                alias.name if not alias.asname else f"{alias.name} as {alias.asname}"
                for alias in node.names
            ])
            line = f"from {full_module} import {names}"
            skeleton_lines.append(line)

    # ✅ 巢狀 class/function 處理
    def process_body(body, level):
        for node in body:
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(base) for base in node.bases] if hasattr(ast, "unparse") else [getattr(base, 'id', '') for base in node.bases]
                base_str = f"({', '.join(bases)})" if bases else ""
                skeleton_lines.append(f"\n{indent(level)}class {node.name}{base_str}:")
                process_body(node.body, level + 1)
            elif isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                args_str = ", ".join(args)
                skeleton_lines.append(f"{indent(level)}def {node.name}({args_str}):")
                skeleton_lines.append(f"{indent(level + 1)}<function_body>")
                process_body(node.body, level + 1)
            elif isinstance(node, ast.AsyncFunctionDef):
                args = [arg.arg for arg in node.args.args]
                args_str = ", ".join(args)
                skeleton_lines.append(f"{indent(level)}async def {node.name}({args_str}):")
                skeleton_lines.append(f"{indent(level + 1)}<function_body>")
                process_body(node.body, level + 1)

    process_body(tree.body, level=0)
    return "\n".join(skeleton_lines)