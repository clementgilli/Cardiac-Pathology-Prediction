from sklearn.base import BaseEstimator, ClassifierMixin
import torch.nn as nn
import torch
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, output_dim=2, lr=1e-3, weight_decay=1e-3, epochs=1000, verbose=False):
        
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.verbose = verbose
        
        in_dim = input_dim
        layers = []
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    def fit(self, X, y):
        torch.manual_seed(0)
        np.random.seed(0)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y-1, dtype=torch.long)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0 and self.verbose:
                print(f"Epoch {epoch} ; Accuracy {(outputs.argmax(dim=1)==batch_y).sum()/len(batch_y)}")
        return self

    def predict(self, X: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs = self.predict_proba(X)
            return probs.argmax(dim=1).cpu().numpy() + 1

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(X), dim=1)

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

class BaggedMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, n_models=30, input_dim=4, output_dim=2,
                 hidden_dim=50, n_layers=1, dropout=0.4,
                 lr=0.008, weight_decay=0.008,
                 feature_fraction=1.0, sample_fraction=0.75,
                 epochs=100, device="cpu", random_state=0):
        self.n_models = n_models
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.feature_fraction = feature_fraction
        self.sample_fraction = sample_fraction
        self.epochs = epochs
        self.device = device
        self.random_state = random_state

    def _create_model(self, input_dim):
        layers = []
        hidden_dim = 32
        in_dim = input_dim

        for _ in range(4):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout)
            ]
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, self.output_dim))
        model = nn.Sequential(GaussianNoise(0.2), *layers)
        return model

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y-1, dtype=torch.long).to(self.device)

        N, D = X.shape
        self.models = []
        self.selected_features = []

        for _ in range(self.n_models):
            indices = np.random.choice(N, int(N * self.sample_fraction), replace=True)
            feat_idx = np.random.choice(D, int(D * self.feature_fraction), replace=False)
            self.selected_features.append(feat_idx)

            x_sub = X[indices][:, feat_idx]
            y_sub = y[indices]

            model = self._create_model(len(feat_idx)).to(self.device)
            self.models.append(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                outputs = model(x_sub)
                loss = criterion(outputs, y_sub)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.predict_proba(X)
            return probs.argmax(dim=1).cpu().numpy() + 1

    def predict_proba(self, X):
        #X = torch.tensor(X, dtype=torch.float32).to(self.device)
        soft_preds = []
        for model, feat_idx in zip(self.models, self.selected_features):
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(X[:, feat_idx]), dim=1)
                soft_preds.append(probs)
        avg_probs = torch.stack(soft_preds).mean(dim=0)
        return avg_probs