"""
Text Mining module for the Grand Lyon Photo Clusters project.
Provides utilities for extracting descriptive terms from photo metadata (tags + titles).

Session 2, Task 4: TF-IDF based cluster descriptions.
Session 3, Task 2: Association rules for cluster naming.
"""

import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from datetime import datetime

# Association rules mining
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

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
    'rhône', 'rhone', 'rhônealpes', 'rhonealpes', 'auvergne', 'auvergnerhonealpes', 'lyons'
}

# Photography and social media stopwords (too generic for cluster naming)
PHOTOGRAPHY_STOPWORDS = {
    'instagram', 'instagramapp', 'iphoneography', 'uploaded', 'square', 'squareformat',
    'iphone', 'iphonesia', 'iphoneonly', 'instagood', 'instadaily', 'instamood',
    'instafood', 'instatravel', 'instaphoto', 'instamoment', 'instalike', 'instafollow',
    'photooftheday', 'picoftheday', 'bestoftheday', 'followme', 'follow', 'like4like',
    'likeforlike', 'tagsforlikes', 'tbt', 'throwbackthursday', 'nofilter', 'filter',
    'vsco', 'vscocam', 'snapseed', 'lightroom', 'photoshop', 'raw', 'hdr',
    'portrait', 'landscape', 'macro', 'closeup', 'wideangle', 'telephoto',
    'street', 'streetphotography', 'urbanphotography', 'travelphotography',
    'foodphotography', 'naturephotography', 'architecturephotography',
    'nikon', 'canon', 'sony', 'fuji', 'fujifilm', 'olympus', 'leica', 'panasonic',
    'dslr', 'mirrorless', 'compact', 'smartphone', 'mobile', 'phonecamera',
    'shot', 'capture', 'shooting', 'taken', 'photography', 'photographer',
    'photo', 'photos', 'picture', 'pictures', 'image', 'images', 'pic', 'pics',
    'flickr', 'flickrphoto', 'explore', 'explored', 'frontpage', 'interestingness',
    'camera', 'lens', 'zoom', 'prime', 'aperture', 'shutter', 'iso', 'exposure',
    'beautiful', 'amazing', 'awesome', 'stunning', 'gorgeous', 'wonderful',
    'love', 'like', 'nice', 'good', 'great', 'best', 'cool', 'cute', 'pretty',
    'travel', 'trip', 'vacation', 'holiday', 'tourism', 'tourist', 'visit',
    'food', 'foodie', 'foodporn', 'yummy', 'delicious', 'tasty', 'eating',
    'asianfood', 'japanesefood', 'koreanfood', 'chinesefood', 'thaifood',
    'nature', 'natural', 'outdoor', 'outdoors', 'outside', 'weather',
    'sky', 'cloud', 'clouds', 'sun', 'sunrise', 'sunset', 'blue', 'red', 'green',
    'night', 'nighttime', 'evening', 'morning', 'afternoon', 'dawn', 'dusk',
    'color', 'colour', 'colors', 'colours', 'colorful', 'colourful',
    'black', 'white', 'blackandwhite', 'bw', 'monochrome', 'sepia',
    'art', 'artistic', 'creative', 'design', 'style', 'fashion',
    'europe', 'european', 'fr', 'city', 'urban', 'town', 'village', 'villages',
    'building', 'buildings', 'architecture', 'structure', 'construction',
    'people', 'person', 'man', 'woman', 'girl', 'boy', 'child', 'children', 'family',
    'sea', 'custom', 'trax', 'pi', 'img', 'dsc', 'jpg', 'jpeg', 'png', 'gif',
    'uploaded', 'uploading', 'upload', 'post', 'share', 'sharing',
    'filoer', 'tourisme', 'villes', 'tours', 'flickrmobile', 'flickriosapp', 'livepic', 'geotagged' # Too generic
}
# Flickrmobile Flickriosapp | Flickriosapp | Livepic Lyons Geotagged

# Combine all stopwords
ALL_STOPWORDS = ENGLISH_STOPWORDS | FRENCH_STOPWORDS | DOMAIN_STOPWORDS | PHOTOGRAPHY_STOPWORDS


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
# ASSOCIATION RULES MINING
# =============================================================================

def get_cluster_transactions(
    df: pd.DataFrame,
    min_terms: int = 2
) -> Dict[int, List[List[str]]]:
    """
    Convert cluster data into transactions for association rules mining.
    Each photo becomes a transaction with its terms as items.
    
    Args:
        df: DataFrame with 'cluster', 'tags', and 'title' columns
        min_terms: Minimum number of terms required for a valid transaction
    
    Returns:
        Dictionary mapping cluster ID to list of transactions (term lists)
    """
    print("Converting cluster data to transactions...")
    
    df = df.copy()
    df['processed_text'] = df.apply(combine_text_fields, axis=1)
    
    cluster_transactions = {}
    
    for cluster_id, group in df.groupby('cluster'):
        if cluster_id == -1:  # Skip noise
            continue
        
        transactions = []
        for text in group['processed_text']:
            if text:
                tokens = text.split()
                # Remove duplicates within a transaction
                unique_tokens = list(dict.fromkeys(tokens))
                if len(unique_tokens) >= min_terms:
                    transactions.append(unique_tokens)
        
        if transactions:
            cluster_transactions[cluster_id] = transactions
    
    print(f"Created transactions for {len(cluster_transactions)} clusters")
    return cluster_transactions


def get_cluster_term_matrix(
    transactions: List[List[str]],
    max_terms: int = 100
) -> pd.DataFrame:
    """
    Convert transactions to a binary term matrix suitable for apriori/fpgrowth.
    
    Args:
        transactions: List of transactions (each is a list of terms)
        max_terms: Maximum number of terms to consider (by frequency)
    
    Returns:
        Binary DataFrame with terms as columns
    """
    if not transactions:
        return pd.DataFrame()
    
    # Get term frequencies to filter
    all_terms = []
    for t in transactions:
        all_terms.extend(t)
    term_counts = Counter(all_terms)
    
    # Keep only top terms
    top_terms = {term for term, _ in term_counts.most_common(max_terms)}
    
    # Filter transactions
    filtered_transactions = [
        [term for term in t if term in top_terms]
        for t in transactions
    ]
    filtered_transactions = [t for t in filtered_transactions if t]
    
    if not filtered_transactions:
        return pd.DataFrame()
    
    # Encode transactions
    te = TransactionEncoder()
    te_array = te.fit_transform(filtered_transactions)
    
    return pd.DataFrame(te_array, columns=te.columns_)


def compute_frequent_itemsets(
    term_matrix: pd.DataFrame,
    min_support: float = 0.05,
    use_fpgrowth: bool = True,
    max_len: int = 3
) -> pd.DataFrame:
    """
    Compute frequent itemsets from a binary term matrix.
    
    Args:
        term_matrix: Binary DataFrame with terms as columns
        min_support: Minimum support threshold (0-1)
        use_fpgrowth: Use FP-Growth (faster) instead of Apriori
        max_len: Maximum itemset length
    
    Returns:
        DataFrame with frequent itemsets and their support
    """
    if term_matrix.empty:
        return pd.DataFrame(columns=['itemsets', 'support'])
    
    try:
        if use_fpgrowth:
            frequent_itemsets = fpgrowth(
                term_matrix,
                min_support=min_support,
                use_colnames=True,
                max_len=max_len
            )
        else:
            frequent_itemsets = apriori(
                term_matrix,
                min_support=min_support,
                use_colnames=True,
                max_len=max_len
            )
        return frequent_itemsets
    except Exception as e:
        print(f"Warning: Itemset mining failed: {e}")
        return pd.DataFrame(columns=['itemsets', 'support'])


def compute_association_rules(
    frequent_itemsets: pd.DataFrame,
    min_confidence: float = 0.5,
    metric: str = 'confidence'
) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.
    
    Args:
        frequent_itemsets: DataFrame from compute_frequent_itemsets
        min_confidence: Minimum confidence threshold
        metric: Metric for filtering ('confidence', 'lift', 'conviction')
    
    Returns:
        DataFrame with association rules
    """
    if frequent_itemsets.empty or len(frequent_itemsets) < 2:
        return pd.DataFrame()
    
    try:
        rules = association_rules(
            frequent_itemsets,
            metric=metric,
            min_threshold=min_confidence,
            num_itemsets=len(frequent_itemsets)
        )
        return rules
    except Exception as e:
        print(f"Warning: Rule generation failed: {e}")
        return pd.DataFrame()


def extract_cluster_rules(
    cluster_transactions: Dict[int, List[List[str]]],
    min_support: float = 0.05,
    min_confidence: float = 0.3,
    top_n_rules: int = 10
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extract association rules for each cluster.
    
    Args:
        cluster_transactions: Dictionary mapping cluster ID to transactions
        min_support: Minimum support for frequent itemsets
        min_confidence: Minimum confidence for rules
        top_n_rules: Maximum number of rules to return per cluster
    
    Returns:
        Dictionary mapping cluster ID to list of rule dictionaries
    """
    print(f"Extracting association rules (support={min_support}, confidence={min_confidence})...")
    
    cluster_rules = {}
    total_rules = 0
    
    for cluster_id, transactions in cluster_transactions.items():
        if len(transactions) < 10:  # Need enough transactions
            cluster_rules[cluster_id] = []
            continue
        
        # Adjust min_support based on cluster size
        adaptive_support = max(0.02, min(0.1, 5 / len(transactions)))
        
        # Build term matrix
        term_matrix = get_cluster_term_matrix(transactions, max_terms=50)
        
        if term_matrix.empty:
            cluster_rules[cluster_id] = []
            continue
        
        # Get frequent itemsets
        itemsets = compute_frequent_itemsets(
            term_matrix,
            min_support=adaptive_support,
            max_len=3
        )
        
        if itemsets.empty:
            cluster_rules[cluster_id] = []
            continue
        
        # Generate rules
        rules = compute_association_rules(itemsets, min_confidence=min_confidence)
        
        if rules.empty:
            # Fallback: use frequent itemsets as descriptors
            top_itemsets = itemsets.nlargest(top_n_rules, 'support')
            cluster_rules[cluster_id] = [
                {
                    'itemset': list(row['itemsets']),
                    'support': row['support'],
                    'type': 'itemset'
                }
                for _, row in top_itemsets.iterrows()
            ]
        else:
            # Sort by lift, then confidence
            rules = rules.sort_values(['lift', 'confidence'], ascending=False)
            top_rules = rules.head(top_n_rules)
            
            cluster_rules[cluster_id] = [
                {
                    'antecedent': list(row['antecedents']),
                    'consequent': list(row['consequents']),
                    'support': row['support'],
                    'confidence': row['confidence'],
                    'lift': row['lift'],
                    'type': 'rule'
                }
                for _, row in top_rules.iterrows()
            ]
        
        total_rules += len(cluster_rules[cluster_id])
    
    print(f"Extracted {total_rules} rules/itemsets across {len(cluster_rules)} clusters")
    return cluster_rules


def get_cluster_itemsets_summary(
    cluster_transactions: Dict[int, List[List[str]]],
    min_support: float = 0.05,
    top_n: int = 5
) -> Dict[int, List[Tuple[frozenset, float]]]:
    """
    Get top frequent itemsets for each cluster (simpler than full rules).
    
    Args:
        cluster_transactions: Dictionary mapping cluster ID to transactions
        min_support: Minimum support threshold
        top_n: Number of top itemsets to return per cluster
    
    Returns:
        Dictionary mapping cluster ID to list of (itemset, support) tuples
    """
    print("Extracting top frequent itemsets per cluster...")
    
    cluster_itemsets = {}
    
    for cluster_id, transactions in cluster_transactions.items():
        if len(transactions) < 5:
            cluster_itemsets[cluster_id] = []
            continue
        
        adaptive_support = max(0.02, min(0.15, 3 / len(transactions)))
        term_matrix = get_cluster_term_matrix(transactions, max_terms=30)
        
        if term_matrix.empty:
            cluster_itemsets[cluster_id] = []
            continue
        
        itemsets = compute_frequent_itemsets(
            term_matrix,
            min_support=adaptive_support,
            max_len=3
        )
        
        if itemsets.empty:
            cluster_itemsets[cluster_id] = []
            continue
        
        # Filter for itemsets with 2+ items (more meaningful)
        multi_item = itemsets[itemsets['itemsets'].apply(len) >= 2]
        if not multi_item.empty:
            top = multi_item.nlargest(top_n, 'support')
        else:
            top = itemsets.nlargest(top_n, 'support')
        
        cluster_itemsets[cluster_id] = [
            (row['itemsets'], row['support'])
            for _, row in top.iterrows()
        ]
    
    return cluster_itemsets


# =============================================================================
# AUTO-NAMING CLUSTERS
# =============================================================================

def is_meaningful_term(term: str) -> bool:
    """
    Check if a term is meaningful for cluster naming.
    Filters out stopwords, short terms, and photography jargon.
    """
    term_lower = term.lower().strip()
    
    # Too short
    if len(term_lower) < 3:
        return False
    
    # Is a stopword
    if term_lower in ALL_STOPWORDS:
        return False
    
    # Check each word in multi-word terms
    words = term_lower.split()
    meaningful_words = [w for w in words if w not in ALL_STOPWORDS and len(w) >= 3]
    
    # At least one meaningful word required
    return len(meaningful_words) > 0


def clean_term_for_name(term: str) -> str:
    """
    Clean a term for use in a cluster name.
    Removes stopwords from multi-word terms.
    """
    words = term.split()
    meaningful_words = [w for w in words if w.lower() not in ALL_STOPWORDS and len(w) >= 2]
    return ' '.join(meaningful_words)


def generate_cluster_name(
    cluster_id: int,
    tfidf_terms: List[Dict[str, Any]] = None,
    rules: List[Dict[str, Any]] = None,
    itemsets: List[Tuple[frozenset, float]] = None,
    max_words: int = 2,
    min_score: float = 0.15
) -> Tuple[str, str]:
    """
    Generate a human-readable name for a cluster.
    
    Priority (CHANGED - TF-IDF first as it's more reliable):
    1. Top TF-IDF terms (most distinctive terms for the cluster)
    2. Frequent itemsets with meaningful terms
    3. Association rules (only if terms are meaningful)
    4. Fallback: "Cluster {id}"
    
    Args:
        cluster_id: Cluster identifier
        tfidf_terms: TF-IDF terms with scores (list of dicts with 'term' and 'score')
        rules: Association rules for this cluster
        itemsets: Frequent itemsets for this cluster
        max_words: Maximum words in the name
        min_score: Minimum TF-IDF score to consider
    
    Returns:
        Tuple of (cluster_name, naming_method)
    """
    # 1. Try TF-IDF terms first (most reliable for distinctive naming)
    if tfidf_terms:
        # Handle both dict format and tuple format
        meaningful_terms = []
        for item in tfidf_terms:
            if isinstance(item, dict):
                term = item.get('term', '')
                score = item.get('score', 0)
            else:
                term, score = item[0], item[1]
            
            if score >= min_score and is_meaningful_term(term):
                cleaned = clean_term_for_name(term)
                if cleaned and cleaned not in meaningful_terms:
                    meaningful_terms.append(cleaned)
        
        if meaningful_terms:
            name_terms = meaningful_terms[:max_words]
            return ' | '.join(t.title() for t in name_terms), 'tfidf'
    
    # 2. Try frequent itemsets (co-occurring meaningful terms)
    if itemsets:
        for itemset, support in itemsets:
            terms = list(itemset)
            meaningful = [t for t in terms if is_meaningful_term(t)]
            if len(meaningful) >= 2:
                name_terms = meaningful[:max_words]
                return ' + '.join(t.title() for t in name_terms), 'frequent_itemset'
    
    # 3. Try association rules (only if both sides are meaningful)
    if rules:
        for rule in rules:
            if rule.get('type') == 'rule' and rule.get('confidence', 0) >= 0.7:
                antecedent = rule.get('antecedent', [])
                consequent = rule.get('consequent', [])
                
                # Get meaningful terms from both sides
                all_terms = antecedent + consequent
                meaningful = [t for t in all_terms if is_meaningful_term(t)]
                
                if len(meaningful) >= 1:
                    name_terms = meaningful[:max_words]
                    return ' + '.join(t.title() for t in name_terms), 'association_rule'
    
    # 4. Fallback
    return f"Cluster {cluster_id}", 'fallback'


def generate_all_cluster_names(
    tfidf_descriptors: Dict[int, List[Tuple[str, float]]] = None,
    cluster_rules: Dict[int, List[Dict[str, Any]]] = None,
    cluster_itemsets: Dict[int, List[Tuple[frozenset, float]]] = None,
    cluster_sizes: Dict[int, int] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Generate names and metadata for all clusters.
    
    Args:
        tfidf_descriptors: TF-IDF descriptors per cluster
        cluster_rules: Association rules per cluster
        cluster_itemsets: Frequent itemsets per cluster
        cluster_sizes: Number of photos per cluster
    
    Returns:
        Dictionary mapping cluster ID to name and metadata
    """
    print("Generating cluster names...")
    
    all_cluster_ids = set()
    if tfidf_descriptors:
        all_cluster_ids.update(tfidf_descriptors.keys())
    if cluster_rules:
        all_cluster_ids.update(cluster_rules.keys())
    if cluster_itemsets:
        all_cluster_ids.update(cluster_itemsets.keys())
    
    cluster_names = {}
    
    for cluster_id in sorted(all_cluster_ids):
        tfidf = tfidf_descriptors.get(cluster_id, []) if tfidf_descriptors else []
        rules = cluster_rules.get(cluster_id, []) if cluster_rules else []
        itemsets = cluster_itemsets.get(cluster_id, []) if cluster_itemsets else []
        size = cluster_sizes.get(cluster_id, 0) if cluster_sizes else 0
        
        # Generate name - now returns (name, method) tuple
        name, method = generate_cluster_name(
            cluster_id=cluster_id,
            tfidf_terms=tfidf,
            rules=rules,
            itemsets=itemsets
        )
        
        # Extract TF-IDF term strings for display
        tfidf_term_list = []
        for item in tfidf[:5]:
            if isinstance(item, dict):
                tfidf_term_list.append(item.get('term', ''))
            else:
                tfidf_term_list.append(item[0] if item else '')
        
        cluster_names[cluster_id] = {
            'name': name,
            'method': method,
            'size': size,
            'top_tfidf_terms': tfidf_term_list,
            'top_itemsets': [
                list(itemset) for itemset, _ in itemsets[:3]
            ] if itemsets else []
        }
    
    # Statistics
    methods = Counter(info['method'] for info in cluster_names.values())
    print(f"Naming methods used: {dict(methods)}")
    
    return cluster_names


# =============================================================================
# ASSOCIATION RULES REPORTING
# =============================================================================

def save_association_rules_json(
    cluster_rules: Dict[int, List[Dict[str, Any]]],
    cluster_names: Dict[int, Dict[str, Any]] = None,
    output_path: Path = None
) -> Path:
    """
    Save association rules to JSON file.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "association_rules.json"
    
    output = {
        "generated_at": datetime.now().isoformat(),
        "method": "FP-Growth + Association Rules",
        "total_clusters": len(cluster_rules),
        "clusters": {}
    }
    
    for cluster_id, rules in cluster_rules.items():
        cluster_data = {
            "rules": rules
        }
        if cluster_names and cluster_id in cluster_names:
            cluster_data["name"] = cluster_names[cluster_id]["name"]
            cluster_data["naming_method"] = cluster_names[cluster_id]["method"]
        
        output["clusters"][str(cluster_id)] = cluster_data
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Saved association rules to {output_path}")
    return output_path


def generate_association_rules_report(
    cluster_rules: Dict[int, List[Dict[str, Any]]],
    cluster_names: Dict[int, Dict[str, Any]],
    df: pd.DataFrame,
    output_path: Path = None
) -> Path:
    """
    Generate a markdown summary report of association rules.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "association_rules_summary.md"
    
    cluster_sizes = df[df['cluster'] != -1].groupby('cluster').size().to_dict()
    
    lines = [
        "# Association Rules Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Method:** FP-Growth for frequent itemsets + Association Rules",
        f"**Total Clusters Analyzed:** {len(cluster_rules)}",
        "",
        "## Methodology",
        "",
        "1. Preprocessed tags and titles for each photo (tokenization, stopword removal)",
        "2. Grouped photos by cluster to create transaction sets",
        "3. Applied FP-Growth algorithm to find frequent term combinations",
        "4. Generated association rules with confidence ≥ 0.3",
        "5. Used rules + itemsets + TF-IDF for automatic cluster naming",
        "",
        "## Cluster Names",
        "",
        "| Cluster | Size | Auto-Generated Name | Method |",
        "|---------|------|---------------------|--------|",
    ]
    
    # Sort by size
    sorted_clusters = sorted(
        cluster_names.keys(),
        key=lambda x: cluster_sizes.get(x, 0),
        reverse=True
    )
    
    for cluster_id in sorted_clusters[:50]:  # Top 50 clusters
        info = cluster_names[cluster_id]
        size = cluster_sizes.get(cluster_id, 0)
        lines.append(f"| {cluster_id} | {size} | {info['name']} | {info['method']} |")
    
    lines.extend([
        "",
        "## Sample Association Rules",
        "",
    ])
    
    # Show detailed rules for top 10 clusters
    for cluster_id in sorted_clusters[:10]:
        info = cluster_names.get(cluster_id, {})
        size = cluster_sizes.get(cluster_id, 0)
        rules = cluster_rules.get(cluster_id, [])
        
        lines.append(f"### Cluster {cluster_id}: {info.get('name', 'Unknown')} ({size} photos)")
        lines.append("")
        
        if not rules:
            lines.append("_No significant patterns found._")
            lines.append("")
            continue
        
        # Show rules or itemsets
        for i, rule in enumerate(rules[:5], 1):
            if rule.get('type') == 'rule':
                ant = ' + '.join(rule['antecedent'])
                cons = ' + '.join(rule['consequent'])
                conf = rule.get('confidence', 0)
                lift = rule.get('lift', 0)
                lines.append(f"{i}. `{ant}` → `{cons}` (conf: {conf:.2f}, lift: {lift:.2f})")
            else:
                items = ' + '.join(rule.get('itemset', []))
                support = rule.get('support', 0)
                lines.append(f"{i}. Frequent itemset: `{items}` (support: {support:.2f})")
        
        lines.append("")
    
    lines.extend([
        "## Naming Method Distribution",
        "",
        "| Method | Count | Percentage |",
        "|--------|-------|------------|",
    ])
    
    method_counts = Counter(info['method'] for info in cluster_names.values())
    total = len(cluster_names)
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        lines.append(f"| {method} | {count} | {pct:.1f}% |")
    
    lines.extend([
        "",
        "## Validation: Known Lyon Landmarks",
        "",
        "| Expected Landmark | Matched Cluster | Auto-Name | Status |",
        "|-------------------|-----------------|-----------|--------|",
    ])
    
    # Look for known landmarks
    known_landmarks = [
        ("Fourvière", ["fourviere", "basilique", "notre dame"]),
        ("Vieux Lyon", ["vieuxlyon", "vieux lyon", "traboule"]),
        ("Place Bellecour", ["bellecour", "place bellecour"]),
        ("Fête des Lumières", ["lumiere", "illuminations", "fete"]),
        ("Demeure du Chaos", ["chaos", "ddc", "abode"]),
        ("Parc Tête d'Or", ["tete or", "zoo", "botanical"]),
        ("Croix-Rousse", ["croixrousse", "croix rousse"]),
        ("Presqu'île", ["presquile", "terreaux", "jacobins"]),
    ]
    
    for landmark_name, keywords in known_landmarks:
        matched = None
        for cluster_id, info in cluster_names.items():
            name_lower = info['name'].lower()
            terms_lower = ' '.join(info.get('top_tfidf_terms', [])).lower()
            if any(kw in name_lower or kw in terms_lower for kw in keywords):
                matched = (cluster_id, info['name'])
                break
        
        if matched:
            lines.append(f"| {landmark_name} | {matched[0]} | {matched[1]} | ✅ |")
        else:
            lines.append(f"| {landmark_name} | - | - | ⚠️ Not found |")
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved association rules report to {output_path}")
    return output_path


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


def run_association_rules_mining(
    df: pd.DataFrame = None,
    min_support: float = 0.05,
    min_confidence: float = 0.3,
    save_results: bool = True
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
    """
    Run association rules mining on clustered data.
    
    Args:
        df: DataFrame with clustered photos (loads from file if None)
        min_support: Minimum support for frequent itemsets
        min_confidence: Minimum confidence for rules
        save_results: Whether to save outputs to files
    
    Returns:
        Tuple of (cluster_rules, cluster_names)
    """
    print("=" * 60)
    print("ASSOCIATION RULES: Mining Cluster Patterns")
    print("=" * 60)
    
    # Load data if not provided
    if df is None:
        data_path = DATA_DIR / "flickr_clustered.csv"
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    
    print(f"Dataset: {len(df)} photos, {df['cluster'].nunique()} clusters")
    
    # Get cluster sizes
    cluster_sizes = df[df['cluster'] != -1].groupby('cluster').size().to_dict()
    
    # Convert to transactions
    cluster_transactions = get_cluster_transactions(df)
    
    # Extract association rules
    cluster_rules = extract_cluster_rules(
        cluster_transactions,
        min_support=min_support,
        min_confidence=min_confidence
    )
    
    # Get frequent itemsets for naming
    cluster_itemsets = get_cluster_itemsets_summary(cluster_transactions)
    
    # Also compute TF-IDF for fallback naming
    print("Computing TF-IDF for naming fallback...")
    cluster_texts = get_cluster_texts(df)
    tfidf_descriptors = compute_tfidf_descriptors(cluster_texts, top_n=10)
    
    # Generate cluster names
    cluster_names = generate_all_cluster_names(
        tfidf_descriptors=tfidf_descriptors,
        cluster_rules=cluster_rules,
        cluster_itemsets=cluster_itemsets,
        cluster_sizes=cluster_sizes
    )
    
    # Save results
    if save_results:
        save_association_rules_json(cluster_rules, cluster_names)
        generate_association_rules_report(cluster_rules, cluster_names, df)
    
    print("=" * 60)
    print("Association rules mining complete!")
    
    return cluster_rules, cluster_names


def run_full_text_mining_pipeline(
    df: pd.DataFrame = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run the complete text mining pipeline: TF-IDF + Association Rules + Auto-naming.
    
    Args:
        df: DataFrame with clustered photos (loads from file if None)
        save_results: Whether to save outputs to files
    
    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("FULL TEXT MINING PIPELINE")
    print("=" * 60)
    
    # Load data if not provided
    if df is None:
        data_path = DATA_DIR / "flickr_clustered.csv"
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    
    # Run TF-IDF
    tfidf_descriptors = run_text_mining(df, save_results=save_results)
    
    # Run Association Rules
    cluster_rules, cluster_names = run_association_rules_mining(df, save_results=save_results)
    
    # Save combined cluster names JSON
    if save_results:
        names_path = REPORTS_DIR / "cluster_names.json"
        with open(names_path, 'w', encoding='utf-8') as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "clusters": {
                    str(cid): info for cid, info in cluster_names.items()
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved cluster names to {names_path}")
    
    print("=" * 60)
    print("Full text mining pipeline complete!")
    print("=" * 60)
    
    return {
        "tfidf_descriptors": tfidf_descriptors,
        "cluster_rules": cluster_rules,
        "cluster_names": cluster_names
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Text mining for cluster descriptions")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top terms per cluster")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    parser.add_argument("--mode", choices=["tfidf", "rules", "full"], default="full",
                        help="Pipeline mode: tfidf, rules, or full")
    
    args = parser.parse_args()
    
    if args.mode == "tfidf":
        descriptors = run_text_mining(
            top_n=args.top_n,
            save_results=not args.no_save
        )
        # Print sample results
        print("\nSample cluster descriptors:")
        for cluster_id in list(descriptors.keys())[:3]:
            terms = [t for t, _ in descriptors[cluster_id][:5]]
            print(f"  Cluster {cluster_id}: {', '.join(terms)}")
    
    elif args.mode == "rules":
        cluster_rules, cluster_names = run_association_rules_mining(
            save_results=not args.no_save
        )
        # Print sample names
        print("\nSample cluster names:")
        for cluster_id in list(cluster_names.keys())[:5]:
            info = cluster_names[cluster_id]
            print(f"  Cluster {cluster_id}: {info['name']} (via {info['method']})")
    
    else:
        results = run_full_text_mining_pipeline(
            save_results=not args.no_save
        )
        # Print sample names
        print("\nSample cluster names:")
        for cluster_id in list(results['cluster_names'].keys())[:5]:
            info = results['cluster_names'][cluster_id]
            print(f"  Cluster {cluster_id}: {info['name']} (via {info['method']})")
