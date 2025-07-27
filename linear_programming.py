import yfinance as yf
import numpy as np
import pulp

tickers = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META',
    'JPM', 'V', 'MA',
    'PG', 'KO', 'PEP',
    'XOM', 'CVX',
    'JNJ', 'UNH'
]
start_date = '2023-01-01'  
end_date = '2025-01-01'   
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
r = data['Adj Close'].pct_change().dropna().to_numpy()  # Macierz zwrotów: r_{i,t} (T × n)
n = len(tickers)  
T = r.shape[0]    
mu = np.mean(r, axis=0)  # Oczekiwane zwroty aktywów (μ_i), średnia z r_{i,t} dla każdego i


print("Oczekiwane dzienne zwroty aktywów (u_i):")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {mu[i]:.6f}")
min_mu = np.min(mu)
max_mu = np.max(mu)
print(f"Zakres dla u_p: [{min_mu:.6f}, {max_mu:.6f}]")

# oczekiwane u_p musi być  w granicach [min(μ_i), max(μ_i)
# ja sobię wyboirę takie:
mu_p = 0.0014

print(f"Wybrano u_p: {mu_p:.6f}")

# Definicja problemu programowania liniowego
problem = pulp.LpProblem("MAD_Portfolio_Optimization", pulp.LpMinimize)

# Zmienne decyzyjne
# w_i: wagi aktywów w portfelu (nieujemne, nie robimy shortow)
w = [pulp.LpVariable(f"w_{i}", lowBound=0, cat='Continuous') for i in range(n)]
# z_t: zmienne pomocnicze reprezentujące odchylenia bezwzględne dla każdego okresu t
z = [pulp.LpVariable(f"z_{t}", lowBound=0, cat='Continuous') for t in range(T)]

# Funkcja celu: minimalizacja MAD = (1/T) * sum(z_t), wzielismy wlasnie MAD bo jest liniowe a nie kwadratowe tak
# jak standardowo w modelu Markowitza, bo nie wtedy musielibysmy rozwazac bardziej zaawansowane programy
problem += (1/T) * pulp.lpSum(z), "Minimalizacja_MAD"


for t in range(T):
    port_return = pulp.lpSum(w[i] * r[t, i] for i in range(n))  # Zwrot portfela w okresie t
    problem += z[t] >= port_return - mu_p, f"z_t_geq_positive_dev_{t}"
    problem += z[t] >= -(port_return - mu_p), f"z_t_geq_negative_dev_{t}"


problem += pulp.lpSum(w) == 1, "Suma_wag"


problem += pulp.lpSum(w[i] * mu[i] for i in range(n)) == mu_p, "Oczekiwany_zwrot"

status = problem.solve()

print("\nStatus:", pulp.LpStatus[status])
print(f"MAD portfela: {pulp.value(problem.objective):.6f}")
print("Wagi portfela:")
for i in range(n):
    print(f"Aktywo {tickers[i]} (w_{i}): {pulp.value(w[i]):.4f}")
portfolio_return = sum(pulp.value(w[i]) * mu[i] for i in range(n))
print(f"Oczekiwany dzienny zwrot portfela (u_p): {portfolio_return:.6f}")
portfolio_returns = np.sum(r * [pulp.value(w[i]) for i in range(n)], axis=1)
mad = np.mean(np.abs(portfolio_returns - mu_p))
print(f"Historyczne MAD portfela: {mad:.6f}")
