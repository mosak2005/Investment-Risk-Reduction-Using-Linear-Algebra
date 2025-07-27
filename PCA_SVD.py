import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


tickers = [

    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META',
    'JPM', 'V', 'MA',
    'PG', 'KO', 'PEP',
    'XOM', 'CVX',
    'JNJ', 'UNH'
]

data = yf.download(tickers, start="2023-06-01", end="2025-06-01", auto_adjust=True)['Close']
returns = data.pct_change().dropna()

# PCA przez SVD na zwrotach
X = returns.values
X_centered = X - np.mean(X, axis=0)

U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
principal_components = Vt
X_pca = X_centered @ principal_components.T

explained_variance = (S ** 2) / (X.shape[0] - 1)
explained_variance_ratio = explained_variance / np.sum(explained_variance)

print("Udział wariancji (na zwrotach):")
for i, ratio in enumerate(explained_variance_ratio[:10]):
    print(f"PC{i+1}: {ratio:.2%}")


plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c='tomato', edgecolor='k')
plt.xlabel('Główna składowa 1')
plt.ylabel('Główna składowa 2')
plt.title('PCA (na dziennych stopach zwrotu)')
plt.grid(True)
plt.tight_layout()
plt.show()





num_components = 4
selected_components = principal_components[:num_components]


weights = selected_components.T @ np.diag(1 / explained_variance[:num_components])
weights = weights.sum(axis=1)

# nie robimy shortow
weights = np.maximum(weights, 0)


weights = weights / weights.sum()


print("Wagi portfela optymalizowanego PCA:")
for ticker, weight in zip(tickers, weights):
    print(f"{ticker}: {weight:.2%}")

equal_weights = np.ones(len(tickers)) / len(tickers)


portfolio_pca = (returns * weights).sum(axis=1)
portfolio_equal = (returns * equal_weights).sum(axis=1)


plt.figure(figsize=(10, 5))
plt.plot(np.cumprod(1 + portfolio_pca), label='Portfel PCA')
plt.plot(np.cumprod(1 + portfolio_equal), label='Portfel równych wag')
plt.title("Porównanie strategii")
plt.xlabel("Dni")
plt.ylabel("Wartość portfela (start = 1)")
plt.legend()
plt.grid()
plt.show()



##### robimy to samo aleee ze shrinkingiem L-W

lambda_ = 0.7
n = X_centered.shape[0]

S = (X_centered.T @ X_centered) / (n - 1)
F = np.diag(np.diag(S))
cov_shrink = (1 - lambda_) * S + lambda_ * F
# PCA weights na podstawie shrinkowanej macierzy
num_components = 4
selected_components = principal_components[:num_components]

weights = selected_components.T @ np.diag(1 / np.diag(cov_shrink)[:num_components])
weights = weights.sum(axis=1)

# nie robimy shortów
weights = np.maximum(weights, 0)
weights = weights / weights.sum()

print("Wagi portfela optymalizowanego PCA:")
for ticker, weight in zip(tickers, weights):
    print(f"{ticker}: {weight:.2%}")

equal_weights = np.ones(len(tickers)) / len(tickers)

portfolio_pca_shrink = (returns * weights).sum(axis=1)
portfolio_equal = (returns * equal_weights).sum(axis=1)

plt.figure(figsize=(10, 5))
plt.plot(np.cumprod(1 + portfolio_pca_shrink), label='Portfel PCA (shrinkage)')
plt.plot(np.cumprod(1 + portfolio_equal), label='Portfel równych wag')
plt.title("Porównanie strategii")
plt.xlabel("Dni")
plt.ylabel("Wartość portfela (start = 1)")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(np.cumprod(1 + portfolio_pca), label='PCA (bez shrinkingu)', linewidth=2)
plt.plot(np.cumprod(1 + portfolio_pca_shrink), label='PCA (z shrinkage L-W)', linewidth=2, linestyle='--')
plt.plot(np.cumprod(1 + portfolio_equal), label='Równe wagi', linewidth=1.5, linestyle=':')
plt.title("Porównanie portfeli: PCA vs PCA z shrinkage")
plt.xlabel("Dni")
plt.ylabel("Wartość portfela (start = 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()