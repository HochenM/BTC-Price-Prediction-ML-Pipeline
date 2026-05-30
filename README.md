
---

## 🎯 Approach

1. **Feature Engineering**: Log returns `log(Close / Close_prev)`
2. **Target**: Next day's log return
3. **Sequencing**: 7-day sliding window → each sample = last 7 days
4. **Train/Test Split**: 85/15 chronologically (no shuffle)
5. **Model**: SGDRegressor with L2 regularization
6. **Conversion**: Log returns → actual dollar prices via `price × exp(log_return)`

---

## 📊 Results

| Metric | Train | Test |
|--------|:-----:|:----:|
| MAE (dollars) | $324 | $683 |
| RMSE (dollars) | — | $1,089 |

**Test MAE ≈ 1.4% of BTC price** — decent for a simple linear model.




## 🧠 Key Concepts Used
Feature Engineering (log returns)

Sliding Window / Sequencing

Time-series Train/Test Split (shuffle=False)

L2 Regularization (Ridge)

SGDRegressor

Baseline Conversion (log → price)

Model Evaluation (MAE, RMSE)

Overfitting Detection

## ⚠️ Limitations
Only one feature (log return)

Linear model — can't capture complex patterns

No transaction costs modeled

Not suitable for real trading without further improvements

## 🔜 Future Improvements
Add features: SMA, volatility, volume, day of week

Try non-linear models: Random Forest, XGBoost

Hyperparameter tuning (GridSearchCV)

Proper time-series cross-validation

Backtesting framework


