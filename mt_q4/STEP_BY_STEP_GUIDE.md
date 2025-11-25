# Adım Adım Rehber (BM25 + T5-small)

1) **Ortamı kur**  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2) **Veriyi indir**  
   ```bash
   python scripts/download_data.py
   ```

3) **Hızlı test** (10 örnek)  
   ```bash
   python -m src.run_experiments --sample-size 10 --top-k 5 --top-n-context 3
   ```

4) **Tam deney** (200 örnek, rapor kaydet)  
   ```bash
   python scripts/run_all_experiments.py
   ```

5) **Sonuçları incele**  
   - Konsoldaki Precision@k / Recall@k / BLEU / ROUGE-L / BERTScore değerlerine bak.  
   - `results_bm25_200.json` içindeki `qualitative_examples` listesinden sadık vs. halüsinatif cevapları etiketle.

6) **Notlar**  
   - Yalnızca BM25 alıcı ve T5-small üretici desteklenir.  
   - Seed 42, deterministik beam search; değiştirmen gerekirse `--seed` ve `--num-beams` argümanlarını kullan.  
   - Daha fazla bağlam için `--top-k` veya `--top-n-context` değerini yükselt, ama `--max-context-chars` sınırını aşmamaya dikkat et.
