# RetailPulse AI — Model Evaluation & Performance Report

**Dataset:** 5 Stores, 15 SKUs (27,375 total sales records across 2023)  
**Holdout Test Period:** Final 28 Days of 2023 (`2023-12-04` to `2023-12-31`, 2,100 test observations)  
**Evaluation Protocol:** Strict temporal split (no data leakage, no shuffling)

---

## 1. Baseline vs. XGBoost Performance

| Model | WAPE | MAE (units) | RMSE (units) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **7-Day Moving Average (Baseline)** | 23.97% | 8.98 | 12.28 | Benchmark |
| **Seasonal Naive (Baseline)** | 24.66% | 9.24 | 12.92 | -2.9% vs MA |
| **XGBoost Regressor (Phase 4)** | **13.60%** | **5.09** | **6.59** | **+43.3% Error Reduction** |

---

## 2. Direct Multi-Horizon Performance (1 to 14 Days Ahead)

Fourteen dedicated direct models ($M_1 \dots M_{14}$) evaluated on the holdout test period to prevent recursive error accumulation:

| Horizon | Target | WAPE | MAE (units) | RMSE (units) |
| :---: | :---: | :---: | :---: | :---: |
| **Day 1** | $t + 1$ | 13.25% | 5.01 | 6.37 |
| **Day 2** | $t + 2$ | 13.21% | 4.99 | 6.34 |
| **Day 3** | $t + 3$ | 13.30% | 5.03 | 6.42 |
| **Day 4** | $t + 4$ | 13.27% | 5.01 | 6.39 |
| **Day 5** | $t + 5$ | 13.27% | 5.01 | 6.41 |
| **Day 6** | $t + 6$ | 13.31% | 5.03 | 6.44 |
| **Day 7** | $t + 7$ | 13.35% | 5.04 | 6.51 |
| **Day 8** | $t + 8$ | 13.14% | 4.96 | 6.40 |
| **Day 9** | $t + 9$ | 13.18% | 4.98 | 6.41 |
| **Day 10** | $t + 10$ | 13.15% | 4.97 | 6.42 |
| **Day 11** | $t + 11$ | 13.09% | 4.94 | 6.33 |
| **Day 12** | $t + 12$ | 13.07% | 4.94 | 6.37 |
| **Day 13** | $t + 13$ | 13.35% | 5.04 | 6.47 |
| **Day 14** | $t + 14$ | 13.42% | 5.07 | 6.48 |
| **Average (14 Days)** | — | **13.24%** | **5.00** | **6.41** |

---

## 3. Quantile Uncertainty & Calibration Summary

Evaluated using **Pinball Loss** with monotonic constraints ($P10 \le P50 \le P90$):

| Metric | Target | Empirical Result | Notes |
| :--- | :---: | :---: | :--- |
| **P10 Pinball Loss** ($\alpha = 0.10$) | — | **1.1715** | Penalizes over-prediction (pessimistic floor) |
| **P50 Pinball Loss** ($\alpha = 0.50$) | — | **2.6578** | Median forecast (MAE proxy: 5.32 units) |
| **P90 Pinball Loss** ($\alpha = 0.90$) | — | **1.2991** | Penalizes stockout risk (safety ceiling) |
| **Mean Pinball Loss** | — | **1.7095** | Overall quantile objective loss |
| **Empirical Coverage ([P10, P90])** | **80.00%** | **84.95%** | Well-calibrated prediction interval |
| **Actuals below P10** | ~10.00% | **8.90%** | Accurate lower tail |
| **Actuals above P90** | ~10.00% | **6.14%** | Conservative buffer preventing stockouts |
| **Average Interval Spread ($P90 - P10$)**| — | **20.07 units** | Dynamic volatility sizing |

---

## 4. Top Feature Drivers (XGBoost Gain Importance)

1. **`rolling_mean_28` (31.75%):** Underlying 4-week sales velocity.
2. **`on_promotion` (16.41%):** Immediate promotional demand surge.
3. **`lag_7` (14.21%):** Day-of-week cyclicality (same day last week).
4. **`lag_14` (10.11%):** Bi-weekly persistence.
5. **`rolling_mean_14` (8.93%):** Medium-term demand momentum.
