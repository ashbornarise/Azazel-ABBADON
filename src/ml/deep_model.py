"""
Rex Clemens — Deep Learning Ensemble
======================================
LSTM + GRU en PyTorch combine avec les classifieurs sklearn.
Prend les features de prix comme input, predit la probabilite de hausse.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


if HAS_TORCH:
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.gru = nn.GRU(hidden_size, hidden_size // 2, 1, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size // 2, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            gru_out, _ = self.gru(lstm_out)
            return self.fc(gru_out[:, -1, :])
else:
    LSTMPredictor = None


class DeepEnsemble:
    def __init__(self, lookback=30, n_features=10, epochs=50, lr=0.001):
        self.lookback = lookback
        self.n_features = n_features
        self.epochs = epochs
        self.lr = lr
        self.scaler = StandardScaler()
        self.deep_model = None
        self.sklearn_models = {}
        self.weights = {"deep": 0.35, "sklearn": 0.65}
        self.trained = False

    def _build_features(self, prices):
        if len(prices) < self.lookback + 30:
            return None, None
        returns = np.diff(np.log(prices))
        features = []
        labels = []
        for i in range(self.lookback, len(returns) - 1):
            window = returns[i - self.lookback:i]
            sma5 = np.mean(prices[i - 4:i + 1])
            sma20 = np.mean(prices[i - 19:i + 1]) if i >= 19 else sma5
            std = np.std(window)
            momentum = np.sum(window[-5:])
            vol_trend = np.mean(np.abs(window[-5:])) / max(np.mean(np.abs(window)), 1e-8)
            rsi = self._calc_rsi(prices[:i + 1])
            feat = [
                window[-1],
                np.mean(window[-5:]),
                np.mean(window[-10:]),
                std,
                momentum,
                prices[i] / sma5 - 1,
                prices[i] / sma20 - 1,
                vol_trend,
                rsi / 100,
                np.max(window) - np.min(window),
            ]
            features.append(feat)
            labels.append(1 if returns[i] > 0 else 0)
        return np.array(features), np.array(labels)

    def train(self, prices):
        features, labels = self._build_features(prices)
        if features is None or len(features) < 100:
            return False

        split = int(len(features) * 0.8)
        X_train, X_test = features[:split], features[split:]
        y_train, y_test = labels[:split], labels[split:]

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.sklearn_models = {
            "RF": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            "GBM": GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42),
        }
        if HAS_LGB:
            self.sklearn_models["LGBM"] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, verbose=-1, random_state=42
            )
        if HAS_XGB:
            self.sklearn_models["XGB"] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, verbosity=0, random_state=42,
                use_label_encoder=False, eval_metric="logloss"
            )

        sk_probas = []
        for name, model in self.sklearn_models.items():
            model.fit(X_train_s, y_train)
            p = model.predict_proba(X_test_s)[:, 1]
            acc = accuracy_score(y_test, (p > 0.5).astype(int))
            sk_probas.append(p)

        self.sk_ensemble_proba = np.mean(sk_probas, axis=0) if sk_probas else None

        if HAS_TORCH:
            try:
                self._train_deep(X_train_s, y_train, X_test_s, y_test)
            except Exception:
                self.deep_model = None

        self.trained = True
        return True

    def _train_deep(self, X_train, y_train, X_test, y_test):
        X_seq_train = self._make_sequences(X_train)
        y_seq_train = torch.FloatTensor(y_train[self.lookback - 1:]).unsqueeze(1)

        self.deep_model = LSTMPredictor(
            input_size=X_train.shape[1],
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        )
        optimizer = torch.optim.Adam(self.deep_model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.deep_model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = self.deep_model(X_seq_train)
            loss = criterion(pred, y_seq_train[:len(pred)])
            loss.backward()
            optimizer.step()

        self.deep_model.eval()

    def _make_sequences(self, X, seq_len=None):
        seq_len = seq_len or min(self.lookback, 10)
        sequences = []
        for i in range(seq_len - 1, len(X)):
            sequences.append(X[i - seq_len + 1:i + 1])
        return torch.FloatTensor(np.array(sequences))

    def predict(self, prices):
        if not self.trained:
            return 0.5

        features, _ = self._build_features(prices)
        if features is None or len(features) == 0:
            return 0.5

        last_feat = features[-1:]
        X_s = self.scaler.transform(last_feat)

        sk_probas = []
        for name, model in self.sklearn_models.items():
            try:
                p = model.predict_proba(X_s)[:, 1][0]
                sk_probas.append(p)
            except Exception:
                pass

        sk_avg = np.mean(sk_probas) if sk_probas else 0.5

        deep_prob = 0.5
        if HAS_TORCH and self.deep_model is not None:
            try:
                self.deep_model.eval()
                with torch.no_grad():
                    seq = self._make_sequences(X_s, seq_len=1)
                    deep_prob = self.deep_model(seq).item()
            except Exception:
                deep_prob = sk_avg

        if self.deep_model is not None:
            final = self.weights["deep"] * deep_prob + self.weights["sklearn"] * sk_avg
        else:
            final = sk_avg

        return round(final, 4)

    def get_model_breakdown(self, prices):
        if not self.trained:
            return {}

        features, _ = self._build_features(prices)
        if features is None or len(features) == 0:
            return {}

        last_feat = features[-1:]
        X_s = self.scaler.transform(last_feat)

        breakdown = {}
        for name, model in self.sklearn_models.items():
            try:
                p = model.predict_proba(X_s)[:, 1][0]
                breakdown[name] = round(p * 100, 1)
            except Exception:
                pass

        if HAS_TORCH and self.deep_model is not None:
            try:
                self.deep_model.eval()
                with torch.no_grad():
                    seq = self._make_sequences(X_s, seq_len=1)
                    breakdown["LSTM+GRU"] = round(self.deep_model(seq).item() * 100, 1)
            except Exception:
                pass

        return breakdown

    @staticmethod
    def _calc_rsi(prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)[-period:]
        gains = np.mean(np.where(deltas > 0, deltas, 0))
        losses = np.mean(np.where(deltas < 0, -deltas, 0))
        if losses == 0:
            return 100.0
        return 100 - 100 / (1 + gains / losses)
