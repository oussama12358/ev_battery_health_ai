import sys
from pathlib import Path
# ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import train_dl
import traceback

print('Attempting to load SOH model...')
try:
    m = train_dl.load_lstm_model('soh', n_features=38, seq_len=10)
    print('Loaded model:', getattr(m, 'name', str(type(m))))
except Exception as e:
    print('Exception:')
    traceback.print_exc()
