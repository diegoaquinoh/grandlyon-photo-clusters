"""
Text Mining module for the Grand Lyon Photo Clusters project.
Provides utilities for extracting descriptive terms from photo metadata (tags + titles).

Session 2, Task 4: TF-IDF based cluster descriptions.
"""

import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

from .data_loader import PROJECT_ROOT

# Output paths
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# STOPWORDS
# =============================================================================

# Common English stopwords
ENGLISH_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we',
    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
    'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
    'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
    'there', 'then', 'once', 'if', 'about', 'after', 'before', 'above', 'below',
    'between', 'under', 'again', 'further', 'because', 'while', 'during', 'until',
}

# Common French stopwords
FRENCH_STOPWORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'au', 'aux', 'ce', 'cette',
    'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre',
    'nos', 'votre', 'vos', 'leur', 'leurs', 'je', 'tu', 'il', 'elle', 'nous',
    'vous', 'ils', 'elles', 'on', 'qui', 'que', 'quoi', 'dont', 'où', 'et', 'ou',
    'mais', 'donc', 'car', 'ni', 'si', 'ne', 'pas', 'plus', 'moins', 'très',
    'bien', 'mal', 'peu', 'trop', 'assez', 'beaucoup', 'toujours', 'jamais',
    'souvent', 'parfois', 'encore', 'déjà', 'avec', 'sans', 'pour', 'par', 'sur',
    'sous', 'dans', 'entre', 'vers', 'chez', 'avant', 'après', 'pendant', 'depuis',
    'être', 'avoir', 'faire', 'aller', 'venir', 'voir', 'pouvoir', 'vouloir',
    'devoir', 'falloir', 'est', 'sont', 'était', 'été', 'étaient', 'ai', 'as', 'a',
    'avons', 'avez', 'ont', 'fait', 'va', 'vont', 'peut', 'peuvent', 'veut',
    'veulent', 'doit', 'doivent', 'faut', 'en', 'y',
}

# Domain-specific stopwords (too common in Flickr/Lyon context)
DOMAIN_STOPWORDS = {
    'lyon', 'france', 'french', 'photo', 'photos', 'picture', 'pictures',
    'image', 'images', 'flickr', 'img', 'dsc', 'jpg', 'jpeg', 'raw',
    'canon', 'nikon', 'sony', 'iphone', 'samsung', 'camera',
    '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020',
    '365', '365project', 'day', 'days', 'january', 'february', 'march', 'april',
    'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'rhône', 'rhone', 'rhônealpes', 'rhonealpes', 'auvergne', 'auvergnerhonealpes',
}

# Combine all stopwords
ALL_STOPWORDS = ENGLISH_STOPWORDS | FRENCH_STOPWORDS | DOMAIN_STOPWORDS


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean a single text string by removing special characters and normalizing.
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace underscores and hyphens with spaces (common in tags)
    text = text.replace('_', ' ').replace('-', ' ')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove special characters except spaces and commas
    text = re.sub(r'[^\w\s,]', ' ', text)
    
    # Remove numbers (usually not meaningful)
    text = re.sub(r'\b\d+\b', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into individual words.
    Handles comma-separated tags and space-separated words.
    
    Args:
        text: Cleaned text string
    
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    # Split by commas first (for tags), then by spaces
    tokens = []
    for part in text.split(','):
        part = part.strip()
        if ' ' in part:
            tokens.extend(part.split())
        elif part:
            tokens.append(part)
    
    # Filter short tokens (less than 2 chars) and stopwords
    tokens = [
        t for t in tokens 
        if len(t) >= 2 and t not in ALL_STOPWORDS
    ]
    
    return tokens


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline: clean + tokenize + rejoin.
    Suitable for TF-IDF input.
    
    Args:
        text: Raw text string
    
    Returns:
        Preprocessed text as space-separated tokens
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    return ' '.join(tokens)


def combine_text_fields(row: pd.Series) -> str:
    """
    Combine tags and title fields from a DataFrame row.
    
    Args:
        row: DataFrame row with 'tags' and 'title' columns
    
    Returns:
        Combined preprocessed text
    """
    parts = []
    
    if pd.notna(row.get('tags')):
        parts.append(str(row['tags']))
    
    if pd.notna(row.get('title')):
        parts.append(str(row['title']))
    
    combined = ', '.join(parts)
    return preprocess_text(combined)


# =============================================================================
# CLUSTER TEXT AGGREGATION
# =============================================================================

def get_cluster_texts(df: pd.DataFrame) -> Dict[int, str]:
    """
    Aggregate all text for each cluster into a single document.
    
    Args:
        df: DataFrame with 'cluster', 'tags', and 'title' columns
    
    Returns:
        Dictionary mapping cluster ID to aggregated text
    """
    print("Preprocessing text for all photos...")
    
    # Preprocess text for each row
    df = df.copy()
    df['processed_text'] = df.apply(combine_text_fields, axis=1)
    
    # Group by cluster and concatenate
    cluster_texts = {}
    for cluster_id, group in df.groupby('cluster'):
        # Skip noise cluster (-1) if present
        if cluster_id == -1:
            continue
        
        texts = group['processed_text'].tolist()
        combined = ' '.join([t for t in texts if t])
        cluster_texts[cluster_id] = combined
    
    print(f"Aggregated text for {len(cluster_texts)} clusters")
    return cluster_texts


# =============================================================================
# TF-IDF ANALYSIS
# =============================================================================

def compute_tfidf_descriptors(
    cluster_texts: Dict[int, str],
    top_n: int = 10,
    min_df: int = 2,
    max_df: float = 0.8
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Compute TF-IDF descriptors for each cluster.
    
    Args:
        cluster_texts: Dictionary mapping cluster ID to aggregated text
        top_n: Number of top terms to extract per cluster
        min_df: Minimum document frequency (ignore rare terms)
        max_df: Maximum document frequency ratio (ignore too common terms)
    
    Returns:
        Dictionary mapping cluster ID to list of (term, score) tuples
    """
    print(f"Computing TF-IDF (top {top_n} terms per cluster)...")
    
    # Prepare documents and maintain cluster ID mapping
    cluster_ids = sorted(cluster_texts.keys())
    documents = [cluster_texts[cid] for cid in cluster_ids]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=5000,
        ngram_range=(1, 2),  # Include bigrams for more context
        token_pattern=r'(?u)\b[a-zA-Z]{2,}\b'  # Only words, min 2 chars
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as e:
        print(f"Warning: TF-IDF failed with error: {e}")
        print("Trying with relaxed parameters...")
        vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=1.0,
            max_features=5000
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract top terms for each cluster
    descriptors = {}
    for idx, cluster_id in enumerate(cluster_ids):
        # Get TF-IDF scores for this cluster
        scores = tfidf_matrix[idx].toarray().flatten()
        
        # Get indices of top terms
        top_indices = scores.argsort()[-top_n:][::-1]
        
        # Build list of (term, score) tuples
        top_terms = [
            (feature_names[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]
        
        descriptors[cluster_id] = top_terms
    
    print(f"Generated descriptors for {len(descriptors)} clusters")
    return descriptors


def get_simple_term_frequencies(
    cluster_texts: Dict[int, str],
    top_n: int = 10
) -> Dict[int, List[Tuple[str, int]]]:
    """
    Get simple term frequencies per cluster (baseline/alternative to TF-IDF).
    
    Args:
        cluster_texts: Dictionary mapping cluster ID to aggregated text
        top_n: Number of top terms to extract per cluster
    
    Returns:
        Dictionary mapping cluster ID to list of (term, count) tuples
    """
    term_freqs = {}
    
    for cluster_id, text in cluster_texts.items():
        tokens = text.split()
        counter = Counter(tokens)
        top_terms = counter.most_common(top_n)
        term_freqs[cluster_id] = top_terms
    
    return term_freqs


# =============================================================================
# OUTPUT / REPORTING
# =============================================================================

def format_descriptors_for_display(
    descriptors: Dict[int, List[Tuple[str, float]]],
    top_n: int = 5
) -> Dict[int, List[str]]:
    """
    Format descriptors for simple display (just term names).
    
    Args:
        descriptors: Full descriptors with scores
        top_n: Number of terms to include
    
    Returns:
        Dictionary mapping cluster ID to list of term names
    """
    return {
        cluster_id: [term for term, _ in terms[:top_n]]
        for cluster_id, terms in descriptors.items()
    }


def save_descriptors_json(
    descriptors: Dict[int, List[Tuple[str, float]]],
    output_path: Path = None
) -> Path:
    """
    Save descriptors to JSON file.
    
    Args:
        descriptors: TF-IDF descriptors
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = REPORTS_DIR / "cluster_descriptors.json"
    
    # Convert to serializable format
    output = {
        "generated_at": datetime.now().isoformat(),
        "method": "TF-IDF",
        "clusters": {
            str(cluster_id): [
                {"term": term, "score": round(score, 4)}
                for term, score in terms
            ]
            for cluster_id, terms in descriptors.items()
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Saved descriptors to {output_path}")
    return output_path


def generate_summary_report(
    descriptors: Dict[int, List[Tuple[str, float]]],
    df: pd.DataFrame,
    output_path: Path = None
) -> Path:
    """
    Generate a markdown summary report of cluster descriptors.
    
    Args:
        descriptors: TF-IDF descriptors
        df: Original DataFrame with cluster info
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = REPORTS_DIR / "text_mining_summary.md"
    
    # Calculate cluster sizes
    cluster_sizes = df[df['cluster'] != -1].groupby('cluster').size()
    
    lines = [
        "# Cluster Text Analysis Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Method:** TF-IDF with bigrams",
        f"**Total Clusters:** {len(descriptors)}",
        "",
        "## Methodology",
        "",
        "- Combined tags and titles for each photo",
        "- Applied stopword filtering (English + French + domain-specific)",
        "- Used TF-IDF to identify distinctive terms per cluster",
        "- Extracted unigrams and bigrams (1-2 word phrases)",
        "",
        "## Cluster Descriptors",
        "",
        "| Cluster | Size | Top Terms |",
        "|---------|------|-----------|",
    ]
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(
        descriptors.keys(),
        key=lambda x: cluster_sizes.get(x, 0),
        reverse=True
    )
    
    for cluster_id in sorted_clusters:
        terms = descriptors[cluster_id]
        size = cluster_sizes.get(cluster_id, 0)
        
        # Get top 5 terms for table
        top_terms = [t for t, _ in terms[:5]]
        terms_str = ", ".join(top_terms) if top_terms else "(no distinctive terms)"
        
        lines.append(f"| {cluster_id} | {size} | {terms_str} |")
    
    lines.extend([
        "",
        "## Sample Cluster Details",
        "",
    ])
    
    # Show detailed info for top 5 clusters
    for cluster_id in sorted_clusters[:5]:
        terms = descriptors[cluster_id]
        size = cluster_sizes.get(cluster_id, 0)
        
        lines.append(f"### Cluster {cluster_id} ({size} photos)")
        lines.append("")
        lines.append("| Rank | Term | TF-IDF Score |")
        lines.append("|------|------|--------------|")
        
        for rank, (term, score) in enumerate(terms[:10], 1):
            lines.append(f"| {rank} | {term} | {score:.4f} |")
        
        lines.append("")
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved summary report to {output_path}")
    return output_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_text_mining(
    df: pd.DataFrame = None,
    top_n: int = 10,
    save_results: bool = True
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Run the complete text mining pipeline on clustered data.
    
    Args:
        df: DataFrame with clustered photos (loads from file if None)
        top_n: Number of top terms per cluster
        save_results: Whether to save outputs to files
    
    Returns:
        Dictionary of cluster descriptors
    """
    print("=" * 60)
    print("TEXT MINING: Generating Cluster Descriptors")
    print("=" * 60)
    
    # Load data if not provided
    if df is None:
        data_path = DATA_DIR / "flickr_clustered.csv"
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    
    print(f"Dataset: {len(df)} photos, {df['cluster'].nunique()} clusters")
    
    # Aggregate text by cluster
    cluster_texts = get_cluster_texts(df)
    
    # Compute TF-IDF descriptors
    descriptors = compute_tfidf_descriptors(cluster_texts, top_n=top_n)
    
    # Save results
    if save_results:
        save_descriptors_json(descriptors)
        generate_summary_report(descriptors, df)
    
    print("=" * 60)
    print("Text mining complete!")
    
    return descriptors


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text mining for cluster descriptions")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top terms per cluster")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    
    args = parser.parse_args()
    
    descriptors = run_text_mining(
        top_n=args.top_n,
        save_results=not args.no_save
    )
    
    # Print sample results
    print("\nSample cluster descriptors:")
    for cluster_id in list(descriptors.keys())[:3]:
        terms = [t for t, _ in descriptors[cluster_id][:5]]
        print(f"  Cluster {cluster_id}: {', '.join(terms)}")
