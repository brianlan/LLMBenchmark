import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
