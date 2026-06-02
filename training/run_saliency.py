import pandas as pd
from pathlib import Path
from training import train_dl
from utils.feature_engineering import build_lstm_sequences
from utils.explainability import lstm_gradient_saliency

ROOT_DIR = Path(__file__).resolve().parent.parent
feat = pd.read_parquet(ROOT_DIR / 'data/processed/cycle_features.parquet')
X_seq, y_soh_seq, y_rul_seq, groups = build_lstm_sequences(feat, seq_len=train_dl.SEQ_LEN, fit_scaler=False)
print('X_seq', X_seq.shape)
model = train_dl.load_lstm_model('soh', n_features=X_seq.shape[2], seq_len=X_seq.shape[1])
print('Model loaded:', model.name)
sal = lstm_gradient_saliency(model, X_seq, sample_idx=0, target='soh', save=True)
print('Saliency shape:', sal.shape)
