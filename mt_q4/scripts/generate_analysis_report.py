import argparse
import json
from typing import Dict, List
import numpy as np

def load_results(filepath: str) -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_faithful_vs_hallucinated(results: Dict) -> Dict:
        qualitative = results.get("qualitative_examples", [])
    
    # If the results are from comparison script
    if "bm25_results" in results:
        qualitative_bm25 = results["bm25_results"].get("qualitative_examples", [])
        qualitative_dense = results["dense_results"].get("qualitative_examples", [])
        qualitative = qualitative_bm25 + qualitative_dense
    
    faithful_examples = [ex for ex in qualitative if ex.get("is_faithful", False)]
    hallucinated_examples = [ex for ex in qualitative if not ex.get("is_faithful", False)]
    
    return {
        "total_examples": len(qualitative),
        "faithful_count": len(faithful_examples),
        "hallucinated_count": len(hallucinated_examples),
        "faithful_ratio": len(faithful_examples) / len(qualitative) if qualitative else 0,
        "faithful_examples": faithful_examples[:3],
        "hallucinated_examples": hallucinated_examples[:3],
    }

def compute_correlation_analysis(results: Dict) -> Dict:
        # For comparison results
    if "bm25_results" in results:
        bm25_r = results["bm25_results"]["retrieval_metrics"]
        bm25_g = results["bm25_results"]["generation_metrics"]
        dense_r = results["dense_results"]["retrieval_metrics"]
        dense_g = results["dense_results"]["generation_metrics"]
        
        retrieval_scores = [
            bm25_r["recall_at_k"],
            dense_r["recall_at_k"]
        ]
        generation_scores = [
            bm25_g["bertscore_f1"],
            dense_g["bertscore_f1"]
        ]
        
        correlation = np.corrcoef(retrieval_scores, generation_scores)[0, 1]
        
        return {
            "retrieval_quality_vs_generation_quality": {
                "bm25": {
                    "recall": bm25_r["recall_at_k"],
                    "bertscore": bm25_g["bertscore_f1"]
                },
                "dense": {
                    "recall": dense_r["recall_at_k"],
                    "bertscore": dense_g["bertscore_f1"]
                },
                "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            }
        }
    else:
        # For single retriever results
        return {
            "note": "Correlation analysis requires comparison data from multiple retrievers"
        }

def generate_markdown_report(
    results: Dict, 
    analysis: Dict, 
    correlation: Dict, 
    output_path: str
):
        report_lines = [
        "# RAG System Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"This report analyzes the interplay between retrieval quality, factual accuracy, "
        f"and generation fluency in a Retrieval-Augmented Generation (RAG) system.",
        "",
    ]
    
    # Add retriever comparison if available
    if "bm25_results" in results:
        report_lines.extend([
            "## Retriever Comparison",
            "",
            "### BM25 (Sparse) Retriever",
            "",
            "**Retrieval Metrics:**",
        ])
        for k, v in results["bm25_results"]["retrieval_metrics"].items():
            report_lines.append(f"- {k}: {v:.4f}")
        
        report_lines.extend([
            "",
            "**Generation Metrics:**",
        ])
        for k, v in results["bm25_results"]["generation_metrics"].items():
            report_lines.append(f"- {k}: {v:.4f}")
        
        report_lines.extend([
            "",
            "### Dense (Sentence-BERT) Retriever",
            "",
            "**Retrieval Metrics:**",
        ])
        for k, v in results["dense_results"]["retrieval_metrics"].items():
            report_lines.append(f"- {k}: {v:.4f}")
        
        report_lines.extend([
            "",
            "**Generation Metrics:**",
        ])
        for k, v in results["dense_results"]["generation_metrics"].items():
            report_lines.append(f"- {k}: {v:.4f}")
        
        report_lines.extend([
            "",
            "### Performance Difference (Dense - BM25)",
            "",
        ])
        for metric_type in ["retrieval_improvement", "generation_improvement"]:
            report_lines.append(f"**{metric_type.replace('_', ' ').title()}:**")
            for k, v in results["comparison"][metric_type].items():
                sign = "+" if v >= 0 else ""
                report_lines.append(f"- {k}: {sign}{v:.4f}")
            report_lines.append("")
    else:
        # Single retriever results
        report_lines.extend([
            "## Retrieval Performance",
            "",
        ])
        for k, v in results.get("retrieval_metrics", {}).items():
            report_lines.append(f"- {k}: {v:.4f}")
        
        report_lines.extend([
            "",
            "## Generation Performance",
            "",
        ])
        for k, v in results.get("generation_metrics", {}).items():
            report_lines.append(f"- {k}: {v:.4f}")
    
    # Analysis section
    report_lines.extend([
        "",
        "## Faithful vs. Hallucinated Generations",
        "",
        f"- **Total Examples Analyzed:** {analysis['total_examples']}",
        f"- **Faithful Generations:** {analysis['faithful_count']} ({analysis['faithful_ratio']:.1%})",
        f"- **Hallucinated Generations:** {analysis['hallucinated_count']} ({1-analysis['faithful_ratio']:.1%})",
        "",
        "**Definition:** A generation is considered 'faithful' if at least one relevant document "
        "was successfully retrieved in the top-k results.",
        "",
    ])
    
    # Faithful examples
    if analysis["faithful_examples"]:
        report_lines.extend([
            "### Examples of Faithful Generations",
            "",
        ])
        for i, ex in enumerate(analysis["faithful_examples"], 1):
            report_lines.extend([
                f"#### Example {i}",
                "",
                f"**Question:** {ex['question']}",
                "",
                f"**Reference Answer:** {ex['reference']}",
                "",
                f"**Generated Answer:** {ex['prediction']}",
                "",
                f"**Retrieved Documents:** {', '.join(ex['context_titles'])}",
                "",
                f"**Relevant Documents:** {', '.join(ex['relevant_titles'])}",
                "",
                "**Analysis:** This generation is considered faithful because relevant documents "
                "were successfully retrieved, providing factual grounding for the answer.",
                "",
            ])
    
    # Hallucinated examples
    if analysis["hallucinated_examples"]:
        report_lines.extend([
            "### Examples of Hallucinated Generations",
            "",
        ])
        for i, ex in enumerate(analysis["hallucinated_examples"], 1):
            report_lines.extend([
                f"#### Example {i}",
                "",
                f"**Question:** {ex['question']}",
                "",
                f"**Reference Answer:** {ex['reference']}",
                "",
                f"**Generated Answer:** {ex['prediction']}",
                "",
                f"**Retrieved Documents:** {', '.join(ex['context_titles'])}",
                "",
                f"**Relevant Documents:** {', '.join(ex['relevant_titles'])}",
                "",
                "**Analysis:** This generation is considered hallucinated because no relevant "
                "documents were retrieved. The model generated an answer without proper factual "
                "grounding, leading to potentially incorrect information.",
                "",
            ])
    
    # Correlation analysis
    report_lines.extend([
        "## Interplay Analysis: Retrieval Quality ↔ Generation Quality",
        "",
    ])
    
    if "retrieval_quality_vs_generation_quality" in correlation:
        corr_data = correlation["retrieval_quality_vs_generation_quality"]
        report_lines.extend([
            f"**Correlation coefficient between Recall@k and BERTScore F1:** {corr_data['correlation']:.3f}",
            "",
            "### Key Findings:",
            "",
        ])
        
        if corr_data['correlation'] > 0.7:
            report_lines.append("- **Strong positive correlation:** Higher retrieval recall strongly "
                              "correlates with better generation quality (BERTScore).")
        elif corr_data['correlation'] > 0.3:
            report_lines.append("- **Moderate positive correlation:** Better retrieval tends to "
                              "improve generation quality.")
        else:
            report_lines.append("- **Weak correlation:** Limited data points or other factors may "
                              "influence generation quality beyond retrieval.")
        
        report_lines.extend([
            "",
            "- **Factual Accuracy:** When relevant documents are retrieved (high recall), the "
            "generator has access to correct information, reducing hallucinations.",
            "",
            "- **Generation Fluency:** BERTScore captures semantic similarity and fluency. "
            "Better retrieved context helps the model generate more accurate and contextually "
            "appropriate responses.",
            "",
            "- **Trade-offs:** While dense retrievers may achieve higher recall, they may retrieve "
            "semantically similar but not precisely relevant documents. BM25's exact term matching "
            "can be advantageous for factual queries requiring specific entities or numbers.",
            "",
        ])
    else:
        report_lines.append(correlation.get("note", "Correlation analysis not available."))
    
    # Conclusions
    report_lines.extend([
        "## Conclusions",
        "",
        "### Retrieval Quality Impact",
        "",
        "Higher Precision@k and Recall@k in retrieval directly improve generation quality by:",
        "1. Providing factually accurate context",
        "2. Reducing model hallucination",
        "3. Enabling the generator to ground answers in retrieved evidence",
        "",
        "### Factual Accuracy Considerations",
        "",
        "- **With Relevant Context:** When retrieval succeeds, the generator produces more "
        "factually accurate answers that align with reference answers.",
        "- **Without Relevant Context:** The model often generates plausible-sounding but "
        "incorrect answers (hallucinations), relying on parametric knowledge.",
        "",
        "### Generation Fluency Trade-offs",
        "",
        "- **BLEU/ROUGE:** Measure lexical overlap; may be low even for semantically correct answers.",
        "- **BERTScore:** Better captures semantic similarity; shows stronger correlation with "
        "retrieval quality.",
        "- **Fluency vs. Accuracy:** A fluent response is not necessarily factually accurate. "
        "Retrieval quality is crucial for factual grounding.",
        "",
        "## Recommendations",
        "",
        "1. **Hybrid Retrieval:** Consider combining BM25 and dense retrievers to leverage both "
        "exact matching and semantic similarity.",
        "",
        "2. **Reranking:** Add a reranking stage to improve precision of top-k results.",
        "",
        "3. **Context Verification:** Implement mechanisms to verify if retrieved documents actually "
        "contain answer-relevant information before generation.",
        "",
        "4. **Longer Context:** Experiment with models that can handle longer contexts to include "
        "more retrieved documents.",
        "",
        "5. **Answer Verification:** Add post-generation verification to detect and flag potential "
        "hallucinations.",
        "",
    ])
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed analysis report from RAG experiment results"
    )
    parser.add_argument(
        "--results", 
        required=True,
        help="Path to JSON results file (from run_experiments.py or compare_retrievers.py)"
    )
    parser.add_argument(
        "--output", 
        default="ANALYSIS_REPORT.md",
        help="Output path for markdown report"
    )
    args = parser.parse_args()
    
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    print("Analyzing faithful vs hallucinated generations...")
    analysis = analyze_faithful_vs_hallucinated(results)
    
    print("Computing correlation analysis...")
    correlation = compute_correlation_analysis(results)
    
    print(f"Generating markdown report to {args.output}...")
    generate_markdown_report(results, analysis, correlation, args.output)
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total examples: {analysis['total_examples']}")
    print(f"Faithful: {analysis['faithful_count']} ({analysis['faithful_ratio']:.1%})")
    print(f"Hallucinated: {analysis['hallucinated_count']} ({1-analysis['faithful_ratio']:.1%})")
    
    if "retrieval_quality_vs_generation_quality" in correlation:
        corr = correlation["retrieval_quality_vs_generation_quality"]["correlation"]
        print(f"\nCorrelation (Recall ↔ BERTScore): {corr:.3f}")
    
    print(f"\nDetailed report saved to: {args.output}")

if __name__ == "__main__":
    main()
