## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works)
- Completed mt_q1 with trained GRU+BERT model

### Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```powershell
# Run complete analysis pipeline
python run_analysis.py

# Generate comprehensive report
python generate_report.py
```

**Expected Runtime:**

- Full analysis: ~30-60 minutes (depending on sample size and GPU)
- Report generation: ~1-2 minutes

---

## Configuration

Edit `config.yaml` to customize the analysis:
