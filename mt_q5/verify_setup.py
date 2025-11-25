import sys
from pathlib import Path

def check_python_version():
        print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies():
        print("\nChecking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('captum', 'Captum'),
        ('lime', 'LIME'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm')
    ]
    
    all_installed = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_model_exists():
        print("\nChecking model checkpoint...")
    
    model_path = Path(__file__).parent.parent / "mt_q1" / "experiments" / "gru_bert" / "gru_bert_best.pth"
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model found: {model_path}")
        print(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"✗ Model NOT found at: {model_path}")
        print("  Please train the model in mt_q1 first:")
        print("  cd ../mt_q1")
        print("  python train.py --model gru --embedding bert")
        return False

def check_cuda():
        print("\nChecking CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_directories():
        print("\nChecking directories...")
    
    dirs = ['experiments', 'visualizations', 'results', 'reports', 'src']
    all_exist = True
    
    for dir_name in dirs:
        dir_path = Path(__file__).parent / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - MISSING")
            all_exist = False
    
    return all_exist

def check_config():
        print("\nChecking configuration...")
    
    config_path = Path(__file__).parent / "config.yaml"
    
    if config_path.exists():
        print(f"✓ config.yaml found")
        
        # Try to load it
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"  Model: {config['model']['type']}")
            print(f"  Sample size: {config['data']['sample_size']}")
            return True
        except Exception as e:
            print(f"⚠ Config file has errors: {e}")
            return False
    else:
        print(f"✗ config.yaml NOT found")
        return False

def main():
        print("="*60)
    print("MT Q5: Setup Verification")
    print("="*60)
    
    results = {
        'Python version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Model checkpoint': check_model_exists(),
        'CUDA': check_cuda(),
        'Directories': check_directories(),
        'Configuration': check_config()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for check, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYou're ready to run the analysis:")
        print("  python quick_example.py      # Quick test (2-3 min)")
        print("  python run_analysis.py       # Full analysis (30-60 min)")
        print("  python generate_report.py    # Generate report (1-2 min)")
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running the analysis.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Train model in mt_q1: cd ../mt_q1 && python train.py --model gru --embedding bert")
        print("  3. Check Python version: python --version")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
