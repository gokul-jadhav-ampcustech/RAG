"""
Enterprise RAG Architecture Implementation
==========================================

Pipeline:
  USER QUESTION
      │
      ▼
  Query Normalization
      │
      ▼
  Query Understanding
      │
  ┌───┴───┐
  │       │
 Original  Query Rewrite
  │       │
  └───┬───┘
      ▼
  HYBRID RETRIEVAL
  ┌───┴────┐
  │        │
 Vector   BM25
  │        │
  └───┬────┘
      ▼
   RRF Fusion
      │
      ▼
  Top 20 Candidates
      │
      ▼
  MMR Diversity
      │
      ▼
   Top 8 Chunks
      │
      ▼
  Semantic Cross Encoder Reranker
      │
      ▼
   Top 5 Chunks
      │
      ▼
  Neighbor Expansion
      │
      ▼
  Context Compression
      │
      ▼
  Relevance / Grounding Gate
     /     \\
  FAIL    PASS
   │        │
   ▼        ▼
 Web Search  LLM
   │        │
   └───┬────┘
       ▼
   FINAL ANSWER
"""
import logging
import re
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.services import embeddings, llm, retriever
from app.db.models import Document, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RagResult:
    status: str
    answer: str
    requires_human_approval: bool
    sources: list = field(default_factory=list)
    response_type: str = "document_based"
    confidence: float = 0.0
    query_type: str = "simple"
    retrieval_stats: dict = field(default_factory=dict)


@dataclass
class EnhancedChunk:
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    chunk_text: str
    similarity: float
    keyword_score: float = 0.0
    mmr_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


# ─────────────────────────────────────────────────────────────
# STAGE 1: Query Normalization
# ─────────────────────────────────────────────────────────────

class QueryNormalizer:
    """Handles query normalization."""

    STOP_WORDS = {
        'please', 'can', 'you', 'tell', 'me', 'explain', 'describe',
        'what', 'is', 'how', 'when', 'where', 'who', 'why', 'the', 'a', 'an',
        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'among', 'do', 'does', 'did',
    }

    @staticmethod
    def normalize(query: str) -> str:
        """Clean and normalize the input query."""
        query = re.sub(r'\s+', ' ', query.strip())
        return query.lower().strip()

    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        return [w for w in words if w not in QueryNormalizer.STOP_WORDS]


# ─────────────────────────────────────────────────────────────
# STAGE 2: Query Understanding & Rewrite
# ─────────────────────────────────────────────────────────────

class QueryUnderstanding:
    """Understands query intent and generates rewrites."""

    # Domain expansions
    DOMAIN_EXPANSIONS = {
        's3': 'amazon s3 simple storage service',
        'vpc': 'virtual private cloud amazon vpc',
        'ec2': 'elastic compute cloud amazon ec2',
        'lambda': 'aws lambda serverless',
        'rds': 'relational database service amazon rds',
        'dynamodb': 'amazon dynamodb nosql database',
        'iam': 'identity access management aws',
        'kms': 'key management service aws kms',
        'ml': 'machine learning artificial intelligence',
        'ai': 'artificial intelligence machine learning',
        'nlp': 'natural language processing text',
        'api': 'application programming interface',
    }

    @staticmethod
    def classify_query(query: str) -> str:
        """Classify query type."""
        q = query.lower()
        if any(q.startswith(w) for w in ['what is', 'what are', 'define', 'meaning of']):
            return 'definition'
        if any(w in q for w in ['how to', 'how do', 'steps to', 'process of']):
            return 'procedural'
        if any(w in q for w in ['compare', 'difference between', 'vs', 'versus']):
            return 'comparison'
        if any(w in q for w in ['why', 'reason', 'cause']):
            return 'causal'
        if any(w in q for w in ['list', 'examples', 'types of']):
            return 'enumeration'
        return 'general'

    @staticmethod
    def rewrite_query(query: str, query_type: str) -> List[str]:
        """Generate rewritten query variants."""
        rewrites = [query]
        q = query.lower()

        # Domain-specific expansion (max 1)
        for abbr, expansion in QueryUnderstanding.DOMAIN_EXPANSIONS.items():
            if re.search(r'\b' + abbr + r'\b', q):
                rewrites.append(expansion)
                break

        # Type-based rewrite (max 1)
        if query_type == 'definition':
            term = re.sub(r'^(what is|what are|define|meaning of)\s+', '', q).strip()
            if term:
                rewrites.append(f"{term} definition overview explanation features")
        elif query_type == 'procedural':
            term = re.sub(r'^(how to|how do|steps to|process of)\s+', '', q).strip()
            if term:
                rewrites.append(f"steps process guide {term}")
        elif query_type == 'comparison':
            rewrites.append(f"{query} similarities differences advantages disadvantages")

        # Remove duplicates, preserve order
        seen = set()
        unique = []
        for r in rewrites:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique[:3]  # max 3 variants


# ─────────────────────────────────────────────────────────────
# STAGE 3: Hybrid Retrieval (Vector + BM25)
# ─────────────────────────────────────────────────────────────

class HybridRetriever:
    """Hybrid vector + BM25 keyword search with RRF fusion."""

    RRF_K = 60

    def __init__(self, db: Session):
        self.db = db

    def vector_search(self, query: str, top_k: int = 20) -> List[EnhancedChunk]:
        """Perform vector similarity search."""
        try:
            embedding = embeddings.embed_text(query)
            chunks = retriever.retrieve_relevant_chunks(
                self.db, embedding, top_k=top_k, similarity_threshold=0.15
            )
            return [
                EnhancedChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    filename=c.filename,
                    chunk_index=c.chunk_index,
                    chunk_text=c.chunk_text,
                    similarity=c.similarity,
                )
                for c in chunks
            ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def bm25_search(self, query: str, keywords: List[str], top_k: int = 20) -> List[EnhancedChunk]:
        """BM25-style search via PostgreSQL full-text search (ts_rank uses similar IDF weighting)."""
        if not keywords:
            return []
        try:
            # Sanitize keywords for tsquery
            safe_keywords = [re.sub(r"[^a-zA-Z0-9]", "", kw) for kw in keywords if len(kw) >= 2]
            if not safe_keywords:
                return []
            search_terms = ' | '.join(safe_keywords)
            sql = text("""
                SELECT dc.id, dc.document_id, d.filename, dc.chunk_index, dc.chunk_text,
                       ts_rank_cd(to_tsvector('english', dc.chunk_text),
                                  to_tsquery('english', :terms)) AS rank
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE to_tsvector('english', dc.chunk_text) @@ to_tsquery('english', :terms)
                ORDER BY rank DESC
                LIMIT :top_k
            """)
            rows = self.db.execute(sql, {'terms': search_terms, 'top_k': top_k}).fetchall()
            return [
                EnhancedChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    filename=row[2],
                    chunk_index=row[3],
                    chunk_text=row[4],
                    similarity=0.0,
                    keyword_score=float(row[5]),
                )
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"BM25 search failed, using LIKE fallback: {e}")
            return self._like_fallback(keywords, top_k)

    def _like_fallback(self, keywords: List[str], top_k: int) -> List[EnhancedChunk]:
        """LIKE-based fallback when full-text search fails."""
        if not keywords:
            return []
        conditions, params = [], {}
        for i, kw in enumerate(keywords[:4]):
            conditions.append(f"LOWER(dc.chunk_text) LIKE :kw{i}")
            params[f'kw{i}'] = f'%{kw.lower()}%'
        params['top_k'] = top_k
        sql = text(f"""
            SELECT dc.id, dc.document_id, d.filename, dc.chunk_index, dc.chunk_text
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE {' OR '.join(conditions)}
            LIMIT :top_k
        """)
        try:
            rows = self.db.execute(sql, params).fetchall()
            return [
                EnhancedChunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    filename=row[2],
                    chunk_index=row[3],
                    chunk_text=row[4],
                    similarity=0.0,
                    keyword_score=sum(1 for kw in keywords if kw in row[4].lower()),
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"LIKE fallback search failed: {e}")
            return []

    def rrf_fusion(self, *chunk_lists: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """Reciprocal Rank Fusion across multiple result lists."""
        # Rank maps per list
        all_chunks: Dict[int, EnhancedChunk] = {}
        rrf_scores: Dict[int, float] = defaultdict(float)

        for chunk_list in chunk_lists:
            for rank, chunk in enumerate(chunk_list, start=1):
                if chunk.chunk_id not in all_chunks:
                    all_chunks[chunk.chunk_id] = chunk
                rrf_scores[chunk.chunk_id] += 1.0 / (self.RRF_K + rank)

        for chunk_id, score in rrf_scores.items():
            all_chunks[chunk_id].final_score = score

        fused = sorted(all_chunks.values(), key=lambda c: c.final_score, reverse=True)
        logger.info(f"RRF fusion: {sum(len(l) for l in chunk_lists)} total → {len(fused)} unique")
        return fused


# ─────────────────────────────────────────────────────────────
# STAGE 4: MMR Diversity (Top 20 → Top 8)
# ─────────────────────────────────────────────────────────────

class MMRDiversifier:
    """Maximal Marginal Relevance for diversity."""

    @staticmethod
    def select(chunks: List[EnhancedChunk], top_k: int = 8, lambda_param: float = 0.7) -> List[EnhancedChunk]:
        if not chunks:
            return []

        # Deduplicate by chunk_id keeping highest score
        seen: Dict[int, EnhancedChunk] = {}
        for c in chunks:
            if c.chunk_id not in seen or c.final_score > seen[c.chunk_id].final_score:
                seen[c.chunk_id] = c
        chunks = list(seen.values())

        if len(chunks) <= top_k:
            return chunks

        selected: List[EnhancedChunk] = []
        remaining = chunks.copy()

        # Start with best scoring chunk
        best = max(remaining, key=lambda c: c.final_score)
        selected.append(best)
        remaining.remove(best)

        while remaining and len(selected) < top_k:
            best_mmr, best_chunk = -float('inf'), None
            for cand in remaining:
                relevance = cand.final_score
                redundancy = max(
                    MMRDiversifier._jaccard(cand.chunk_text, sel.chunk_text)
                    for sel in selected
                )
                mmr = lambda_param * relevance - (1 - lambda_param) * redundancy
                cand.mmr_score = mmr
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_chunk = cand
            if best_chunk:
                selected.append(best_chunk)
                remaining.remove(best_chunk)

        logger.info(f"MMR: {len(chunks)} → {len(selected)} diverse chunks")
        return selected

    @staticmethod
    def _jaccard(text1: str, text2: str) -> float:
        w1, w2 = set(text1.lower().split()), set(text2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)


# ─────────────────────────────────────────────────────────────
# STAGE 5: Semantic Cross-Encoder Reranker (Top 8 → Top 5)
# ─────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Semantic reranker using cross-encoder style scoring.
    Uses a combination of:
    - BM25-style TF-IDF term overlap (approximates cross-encoder without heavy model)
    - RRF score
    - Query coverage
    - Positional bonus (earlier chunks in document tend to have intros/summaries)
    """

    @staticmethod
    def rerank(chunks: List[EnhancedChunk], query: str, top_k: int = 5) -> List[EnhancedChunk]:
        if not chunks:
            return []

        query_terms = set(re.findall(r'\b[a-zA-Z]{2,}\b', query.lower()))

        for chunk in chunks:
            chunk_lower = chunk.chunk_text.lower()
            chunk_terms = set(re.findall(r'\b[a-zA-Z]{2,}\b', chunk_lower))

            # Query term coverage
            if query_terms:
                coverage = len(query_terms & chunk_terms) / len(query_terms)
            else:
                coverage = 0.0

            # TF-IDF style: penalize very long chunks (they may be diluted)
            words = chunk.chunk_text.split()
            length_penalty = min(1.0, 200 / max(len(words), 1)) if len(words) > 200 else 1.0

            # Exact phrase match bonus
            phrase_bonus = 0.0
            query_words = query.lower().split()
            for i in range(len(query_words) - 1):
                bigram = f"{query_words[i]} {query_words[i+1]}"
                if bigram in chunk_lower:
                    phrase_bonus += 0.1

            # Positional bonus: earlier chunks often have key information
            pos_bonus = max(0.0, 0.05 - chunk.chunk_index * 0.005)

            # Combined cross-encoder score
            chunk.rerank_score = (
                0.50 * chunk.final_score +      # RRF hybrid score (primary)
                0.25 * coverage +               # Query term coverage
                0.10 * length_penalty +         # Length normalization
                0.10 * min(phrase_bonus, 0.3) + # Exact phrase matching
                0.05 * pos_bonus                # Position bonus
            )
            chunk.final_score = chunk.rerank_score

        reranked = sorted(chunks, key=lambda c: c.final_score, reverse=True)
        logger.info(f"Cross-encoder rerank: {len(chunks)} → top {top_k}")
        return reranked[:top_k]


# ─────────────────────────────────────────────────────────────
# STAGE 6: Neighbor Expansion
# ─────────────────────────────────────────────────────────────

class NeighborExpander:
    """Expands chunks with neighboring context."""
    #,neighbor services will manner 

    def __init__(self, db: Session):
        self.db = db

    def expand(self, chunks: List[EnhancedChunk], window: int = 1) -> List[Dict]:
        """Fetch neighboring chunks and build enriched context blocks."""
        expanded = []
        for chunk in chunks:
            block = self._fetch_window(chunk, window)
            expanded.append(block)
        return expanded

    def _fetch_window(self, chunk: EnhancedChunk, window: int) -> Dict:
        try:
            sql = text("""
                SELECT chunk_text, chunk_index
                FROM document_chunks
                WHERE document_id = :doc_id
                  AND chunk_index BETWEEN :start AND :end
                ORDER BY chunk_index
            """)
            rows = self.db.execute(sql, {
                'doc_id': chunk.document_id,
                'start': max(0, chunk.chunk_index - window),
                'end': chunk.chunk_index + window,
            }).fetchall()

            parts = []
            for text_val, idx in rows:
                if idx == chunk.chunk_index:
                    parts.append(text_val)
                else:
                    parts.append(text_val)

            return {
                'text': "\n\n".join(parts),
                'filename': chunk.filename,
                'chunk_index': chunk.chunk_index,
                'score': chunk.final_score,
            }
        except Exception as e:
            logger.warning(f"Neighbor expansion failed for chunk {chunk.chunk_id}: {e}")
            return {
                'text': chunk.chunk_text,
                'filename': chunk.filename,
                'chunk_index': chunk.chunk_index,
                'score': chunk.final_score,
            }


# ─────────────────────────────────────────────────────────────
# STAGE 7: Context Compression
# ─────────────────────────────────────────────────────────────

class ContextCompressor:
    """Compresses context to fit within token budget."""

    MAX_CHARS = 12000  # ~3000 tokens

    @staticmethod
    def compress(blocks: List[Dict], query: str) -> List[str]:
        """Select and compress context blocks to fit budget."""
        total = 0
        result = []
        query_terms = set(re.findall(r'\b[a-zA-Z]{2,}\b', query.lower()))

        for block in blocks:
            text = block['text']
            # Score sentences by query term relevance
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 6:
                # Keep sentences with highest query term overlap
                scored = []
                for s in sentences:
                    score = sum(1 for t in query_terms if t in s.lower())
                    scored.append((score, s))
                scored.sort(reverse=True)
                # Take top 70% by relevance, preserving some order
                keep_n = max(3, int(len(sentences) * 0.7))
                top_sentences = [s for _, s in scored[:keep_n]]
                text = ' '.join(top_sentences)

            if total + len(text) > ContextCompressor.MAX_CHARS:
                remaining = ContextCompressor.MAX_CHARS - total
                if remaining > 200:
                    text = text[:remaining] + "..."
                else:
                    break

            result.append(f"[Source: {block['filename']}, chunk {block['chunk_index'] + 1}]\n{text}")
            total += len(text)

        logger.info(f"Context compression: {len(blocks)} blocks → {len(result)}, ~{total} chars")
        return result


# ─────────────────────────────────────────────────────────────
# STAGE 8: Relevance / Grounding Gate
# ─────────────────────────────────────────────────────────────

class RelevanceGate:
    """Determines if retrieved chunks are relevant enough."""

    @staticmethod
    def evaluate(chunks: List[EnhancedChunk], threshold: float = 0.012) -> Tuple[bool, float]:
        """
        Returns (passes_gate, best_score).
        Threshold is calibrated for RRF scores (typically 0.008–0.030 range).
        """
        if not chunks:
            return False, 0.0
        best_score = chunks[0].final_score
        passes = best_score >= threshold
        logger.info(f"Relevance gate: score={best_score:.4f}, threshold={threshold}, passes={passes}")
        return passes, best_score


# ─────────────────────────────────────────────────────────────
# STAGE 9: General AI Fallback
# ─────────────────────────────────────────────────────────────

def _general_ai_answer(question: str, memory_context: str = "") -> RagResult:
    """Answer using general LLM knowledge when documents are irrelevant or absent."""
    logger.info("[GENERAL_AI] Switching to general AI mode for: %r", question[:80])
    answer = llm.generate_general_answer(question, memory_context=memory_context)
    return RagResult(
        status="answered",
        answer=answer,
        requires_human_approval=False,
        response_type="general_ai",
        confidence=0.7,
    )


# ─────────────────────────────────────────────────────────────
# STAGE 10: HITL Fallback (only for explicit web search approval)
# ─────────────────────────────────────────────────────────────

def request_hitl_approval(question: str) -> RagResult:
    """
    Return a HITL approval request instead of auto-searching the web.
    The frontend must show the user two choices:
      - search_web
      - documents_only
    """
    logger.info("[HITL] Waiting for user approval — no relevant documents found")
    return RagResult(
        status="awaiting_human_approval",
        answer=(
            "I couldn't find relevant information in your uploaded documents. "
            "Would you like me to search the web for this?"
        ),
        requires_human_approval=True,
        response_type="hitl_pending",
        confidence=0.0,
    )


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def answer_question(db: Session, question: str, memory_context: str = "") -> RagResult:
    """
    Full enterprise RAG pipeline:
    Question → Normalize → Understand → Hybrid Retrieve →
    RRF Fusion → MMR (top 20→8) → Cross-Encoder Rerank (8→5) →
    Neighbor Expand → Compress → Grounding Gate → LLM / Web Search
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    logger.info(f"RAG pipeline start: {question!r}")
    stats: Dict = {}

    # Check document availability — no docs means general AI, not HITL
    doc_count = db.query(Document).count()
    if doc_count == 0:
        logger.info("[GENERAL_AI] No documents in DB -> general AI answer")
        return _general_ai_answer(question, memory_context)

    # ── Stage 1: Query Normalization ──────────────────────────
    normalized = QueryNormalizer.normalize(question)
    keywords = QueryNormalizer.extract_keywords(normalized)
    stats['normalized_query'] = normalized
    stats['keywords'] = keywords
    logger.info(f"Stage 1 | normalized='{normalized}' | keywords={keywords}")

    # ── Stage 2: Query Understanding & Rewrite ────────────────
    query_type = QueryUnderstanding.classify_query(normalized)
    query_variants = QueryUnderstanding.rewrite_query(normalized, query_type)
    stats['query_type'] = query_type
    stats['query_variants'] = len(query_variants)
    logger.info(f"Stage 2 | type={query_type} | variants={query_variants}")

    # ── Stage 3: Hybrid Retrieval ─────────────────────────────
    hybrid = HybridRetriever(db)

    # Vector search across all query variants
    vector_chunks: List[EnhancedChunk] = []
    for variant in query_variants:
        chunks = hybrid.vector_search(variant, top_k=15)
        vector_chunks.extend(chunks)

    # BM25 keyword search
    bm25_chunks = hybrid.bm25_search(normalized, keywords, top_k=20)

    stats['vector_candidates'] = len(vector_chunks)
    stats['bm25_candidates'] = len(bm25_chunks)
    logger.info(f"Stage 3 | vector={len(vector_chunks)}, bm25={len(bm25_chunks)}")

    # ── Stage 3.5: RRF Fusion → Top 20 ───────────────────────
    fused = hybrid.rrf_fusion(vector_chunks, bm25_chunks)
    top20 = fused[:20]
    stats['after_rrf'] = len(top20)
    logger.info(f"Stage 3.5 | RRF fusion → {len(top20)} candidates")

    if not top20:
        logger.info("[GENERAL_AI] No chunks after RRF fusion -> general AI answer")
        return _general_ai_answer(question, memory_context)

    # ── Stage 4: MMR Diversity → Top 8 ───────────────────────
    top8 = MMRDiversifier.select(top20, top_k=8, lambda_param=0.70)
    stats['after_mmr'] = len(top8)
    logger.info(f"Stage 4 | MMR diversity → {len(top8)} chunks")

    # ── Stage 5: Cross-Encoder Rerank → Top 5 ────────────────
    top5 = CrossEncoderReranker.rerank(top8, normalized, top_k=5)
    stats['after_rerank'] = len(top5)
    logger.info(f"Stage 5 | Cross-encoder rerank → {len(top5)} chunks")

    # ── Stage 8: Relevance / Grounding Gate ───────────────────
    passes_gate, best_score = RelevanceGate.evaluate(top5)
    stats['best_score'] = round(best_score, 4)
    stats['passed_gate'] = passes_gate

    if not passes_gate:
        logger.info("[GENERAL_AI] Gate FAIL (score=%.4f) -> general AI answer", best_score)
        return _general_ai_answer(question, memory_context)

    # ── Stage 6: Neighbor Expansionon ───────────────────────────
    expander = NeighborExpander(db)
    expanded_blocks = expander.expand(top5, window=1)
    logger.info(f"Stage 6 | Neighbor expansion: {len(expanded_blocks)} blocks")

    # ── Stage 7: Context Compression ──────────────────────────
    context_chunks = ContextCompressor.compress(expanded_blocks, normalized)
    stats['context_chunks'] = len(context_chunks)
    logger.info(f"Stage 7 | Context compression: {len(context_chunks)} context blocks")

    # ── Stage 9: LLM Generation ───────────────────────────────
    logger.info("Stage 9 | LLM generation")
    try:
        answer = llm.generate_answer(question, context_chunks, memory_context=memory_context)
    except Exception as e:
        logger.error(f"LLM failed: {e} -> general AI fallback")
        return _general_ai_answer(question, memory_context)

    # If LLM could not ground the answer in documents, fall back to general AI
    _NOT_FOUND_SENTINEL = "this information is not available in the uploaded documents"
    if _NOT_FOUND_SENTINEL in answer.lower():
        logger.info("[GENERAL_AI] LLM returned not-found sentinel -> general AI answer")
        return _general_ai_answer(question, memory_context)

    # Build source metadata
    avg_score = sum(c.final_score for c in top5) / len(top5) if top5 else 0.0
    confidence = min(best_score * 30, 1.0)  # RRF scores are small (0.008-0.030)
    sources = [
        {
            'filename': c.filename,
            'chunk_id': c.chunk_id,
            'chunk_index': c.chunk_index,
            'similarity': round(c.similarity, 3),
            'keyword_score': round(c.keyword_score, 3),
            'rrf_score': round(c.final_score, 4),
            'rank': i + 1,
        }
        for i, c in enumerate(top5[:3])
    ]

    stats.update({
        'avg_rrf_score': round(avg_score, 4),
        'confidence': round(confidence, 3),
        'pipeline_stages': 9,
    })

    logger.info(f"RAG pipeline complete | confidence={confidence:.2f}")
    return RagResult(
        status="answered",
        answer=answer,
        requires_human_approval=False,
        sources=sources,
        response_type="document_based",
        confidence=round(confidence, 2),
        query_type=query_type,
        retrieval_stats=stats,
    )