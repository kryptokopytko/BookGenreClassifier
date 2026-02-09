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

for lib in libraries:
    if not install(lib):
        all_ok = False

if all_ok:
    print("\nWszystkie wymagane biblioteki zostały zainstalowane poprawnie 😊")
else:
    print("\n⚠️ Niektóre biblioteki nie zostały zainstalowane poprawnie. Sprawdź powyższe komunikaty.")
