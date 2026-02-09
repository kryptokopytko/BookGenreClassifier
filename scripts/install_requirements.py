import subprocess
import sys

libraries = [
    "argparse",
    "beautifulsoup4",
    "joblib",
    "lightgbm",
    "matplotlib",
    "nltk",
    "numpy",
    "pandas",
    "requests",
    "scipy",
    "seaborn",
    "scikit-learn",
    "stanza",
    "tqdm",
    "xgboost"
]

def install(package):
    try:
        __import__(package.split()[0])
        print(f"Biblioteka {package} jest już zainstalowana ✅")
        return True
    except ImportError:
        print(f"Instalacja {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except subprocess.CalledProcessError:
            print(f"Nie udało się zainstalować {package} ✖️")
            return False

all_ok = True

try:
    import torch
    print("Biblioteka torch jest już zainstalowana ✅")
except ImportError:
    print("Instalacja CPU-only torch...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
    except subprocess.CalledProcessError:
        print("Nie udało się zainstalować torch ✖️")
        all_ok = False

for lib in libraries:
    if not install(lib):
        all_ok = False

if all_ok:
    print("\nWszystkie wymagane biblioteki zostały zainstalowane poprawnie 😊")
else:
    print("\n⚠️ Niektóre biblioteki nie zostały zainstalowane poprawnie. Sprawdź powyższe komunikaty.")
