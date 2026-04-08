# March Machine Learning Mania 2026 — 6th Place Solution

**Final Brier Score:** `"0.1181191"`  
**Final Rank:** 🥇 **6th** / 3,485 teams

---

##  Overview

This repository contains my solution for the **March Machine Learning Mania 2026** Kaggle competition.

The objective is to predict the probability that one NCAA team beats another, evaluated using **Brier Score**  .

My approach relies on a **simple but highly effective ensemble** of three independent modeling pipelines.

---

##  Solution Summary

The final submission is obtained by **averaging predictions** from three diverse models:

| # | Model | Key Features |
|---|-------|---------------|
| 1️ | **Elo-Based Model** | Margin-of-victory adjusted Elo ratings, seasonal decay |
| 2️ | **Four Factors Model** | eFG%, TOV%, ORB%, FTR (Dean Oliver's framework) |
| 3️ | **Context Model** | Late-season momentum, conference strength, coaching experience |

---

##  Final Ensemble

Predictions are combined using **simple averaging**:

```python
final_pred = (pred_elo + pred_ff + sub_pred_ctx1+sub_pred_ctx2+sub_pred_ctx3+sub_pred_ctx4+sub_pred_ctx5+sub_pred_ctx6+sub_pred_ctx7+sub_pred_ctx8) / 10
```
## Post-processing
Applied minor calibration adjustments on select matchups ( women's tournament)
we leveraged 25 human-created brackets (Men’s tournament) as an additional external signal.:
```python

P_final = 0.95 * P_model + 0.05 * P_human
```

Minor manual corrections were applied during post-processing to fix input inconsistencies (e.g., accidental replacement of 0 with 2 due to keyboard input).

```python

final_pred = np.clip(final_pred, 0.005, 0.995)
```
---

## Key Design Choices

- Model diversity over complexity
- Feature engineering > hyperparameter tuning
- Avoided heavy stacking to reduce overfitting
- Focused on stable, well-calibrated probabilities

---

## Validation

- Training data: historical NCAA tournaments
- Validation seasons: 2022–2025
- Metric: Brier Score

The ensemble consistently outperformed individual models in validation.

---

## Failure Analysis (Short)

- Struggled in:
  - Close matchups (similar seeds)
  - High-variance upset games
- Less sensitivity to:
  - Injury/news-based changes
  - Late unexpected form shifts

---

## Reproduction

1. Install Requirements
```python
pip install -r requirements.txt
```
Or manually:
```python
pip install numpy pandas scikit-learn lightgbm xgboost catboost kaggle
```
---

2. Download Data

You need Kaggle API access:
```python
pip install kaggle
kaggle competitions download -c march-machine-learning-mania-2026 -p data/
cd data
unzip "*.zip"
rm *.zip
cd ..
```
---

3. Run Pipelines

Run the three notebooks (or scripts):
```python
notebooks/
├── elo_model.ipynb
├── four_factors_model.ipynb
├── context_model.ipynb
```
Each notebook generates predictions.

---

4. Ensemble Predictions
```python

final_pred = (pred_elo + pred_ff + sub_pred_ctx1+sub_pred_ctx2+sub_pred_ctx3+sub_pred_ctx4+sub_pred_ctx5+sub_pred_ctx6+sub_pred_ctx7+sub_pred_ctx8) / 10```

Apply clipping:
```python

final_pred = np.clip(final_pred, 0.005, 0.995)
```

---

## Repository Structure
```python

.
├── data/
├── notebooks/
├── outputs/
├── submission/
├── SOLUTION_WRITEUP.md
└── README.md
```
---

## Key Takeaways

- Simple averaging of strong models can outperform complex ensembles
- Feature quality matters more than model complexity
- Calibration is critical for Brier Score optimization
- Diversity between models is the main driver of performance

---

## Possible Improvements

- Weighted averaging instead of uniform
- Probability calibration (Isotonic / Platt scaling)
- External ranking systems (Massey Ordinals, KenPom, etc.)
- Separate models for men and women

---

## Acknowledgments

Thanks to the Kaggle community for valuable discussions and shared insights.

---
