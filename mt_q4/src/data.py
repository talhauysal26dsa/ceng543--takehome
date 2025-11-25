from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple
import random

from datasets import load_dataset

@dataclass
class Document:
    doc_id: str
    title: str
    text: str

@dataclass
class QAExample:
    question: str
    answer: str
    relevant_titles: Set[str]

def load_hotpotqa(
    sample_size: int = 200, split: str = "validation", seed: int = 42
) -> Tuple[List[Document], List[QAExample]]:
        random.seed(seed)
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    sample_size = min(sample_size, len(ds))
    indices = random.sample(range(len(ds)), sample_size)
    subset = ds.select(indices)

    doc_by_title: Dict[str, str] = {}
    examples: List[QAExample] = []

    for record in subset:
        titles = record["context"]["title"]
        paragraphs = record["context"]["sentences"]
        for title, sentences in zip(titles, paragraphs):
            if title not in doc_by_title:
                doc_by_title[title] = " ".join(sentences)
        supporting_titles = set(record["supporting_facts"]["title"])
        examples.append(
            QAExample(
                question=record["question"],
                answer=record["answer"],
                relevant_titles=supporting_titles,
            )
        )

    documents = [
        Document(doc_id=str(idx), title=title, text=text)
        for idx, (title, text) in enumerate(doc_by_title.items())
    ]
    return documents, examples

def build_doc_lookup(documents: Sequence[Document]) -> Dict[str, Document]:
    return {doc.doc_id: doc for doc in documents}

def compose_context(
    doc_lookup: Dict[str, Document], doc_ids: Sequence[str], max_chars: int = 2000
) -> str:
        parts: List[str] = []
    total = 0
    for doc_id in doc_ids:
        doc = doc_lookup[doc_id]
        # Cleaner format for better comprehension
        snippet = f"{doc.title}. {doc.text}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return " ".join(parts)

__all__ = [
    "Document",
    "QAExample",
    "load_hotpotqa",
    "build_doc_lookup",
    "compose_context",
]
