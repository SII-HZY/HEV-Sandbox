"""
Data Loading Utilities

Helper functions for loading and processing data files.
"""

import json
from typing import List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} records from {file_path}")
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(data)} records to {file_path}")


def load_json(file_path: str) -> Dict:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded data from {file_path}")
    return data


def save_json(data: Dict, file_path: str, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Output file path
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    logger.info(f"Saved data to {file_path}")


def load_text_corpus(file_path: str, max_lines: int = None) -> List[str]:
    """
    Load text corpus from file.
    
    Args:
        file_path: Path to text file
        max_lines: Maximum number of lines to load
        
    Returns:
        List of text lines
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line:
                lines.append(line)
    
    logger.info(f"Loaded {len(lines)} lines from {file_path}")
    return lines
