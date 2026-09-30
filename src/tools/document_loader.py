import io
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

import pypdf
import docx

logger = logging.getLogger(__name__)

def extract_text_from_file_bytes(file_bytes: bytes, filename: str) -> str:
    """Extracts plain text from file bytes based on file extension (.pdf, .docx, .txt, .md)."""
    ext = Path(filename).suffix.lower()
    
    if ext == ".pdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages_text).strip()
        except Exception as e:
            logger.error(f"Error reading PDF {filename}: {e}")
            return ""
            
    elif ext == ".docx":
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paras).strip()
        except Exception as e:
            logger.error(f"Error reading DOCX {filename}: {e}")
            return ""
            
    elif ext in [".txt", ".md"]:
        try:
            return file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1").strip()
            except Exception as e:
                logger.error(f"Error decoding text file {filename}: {e}")
                return ""
    else:
        logger.warning(f"Unsupported document file format: {ext}")
        return ""

def extract_text_from_path(file_path: Union[str, Path]) -> str:
    """Extracts plain text from a file on the local filesystem."""
    p = Path(file_path)
    if not p.is_file():
        logger.error(f"File not found: {file_path}")
        return ""
    with open(p, "rb") as f:
        data = f.read()
    return extract_text_from_file_bytes(data, p.name)

def chunk_document(text: str, filename: str, chunk_size: int = 900, overlap: int = 150) -> List[Dict[str, Any]]:
    """Splits a document text into overlapping chunks with source metadata."""
    if not text.strip():
        return []
        
    cleaned_text = re.sub(r'\r\n', '\n', text).strip()
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_idx = 1
    
    while start < len(cleaned_text):
        end = start + chunk_size
        chunk_content = cleaned_text[start:end].strip()
        if chunk_content:
            chunks.append({
                "source": filename,
                "chunk_id": chunk_idx,
                "content": chunk_content
            })
            chunk_idx += 1
            
        start += chunk_size - overlap
        
    return chunks

def search_private_docs(query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieves top_k most relevant chunks using weighted term overlap and frequency."""
    if not chunks or not query.strip():
        return []
        
    query_tokens = re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', query.lower())
    if not query_tokens:
        return []
        
    q_token_set = set(query_tokens)
    scored_chunks = []
    
    for c in chunks:
        content = c.get("content", "")
        c_tokens = re.findall(r'\b[a-zA-Z0-9_-]{3,}\b', content.lower())
        if not c_tokens:
            continue
            
        # Compute term frequency matches
        matches = [t for t in c_tokens if t in q_token_set]
        if matches:
            unique_matches = len(set(matches))
            total_matches = len(matches)
            # Score favors both diversity of matched query terms and total frequency
            score = (unique_matches * 2.0) + (total_matches * 0.5)
            scored_chunks.append((score, c))
            
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_chunks[:top_k]]
