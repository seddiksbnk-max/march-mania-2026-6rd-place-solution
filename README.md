🏀 March Machine Learning Mania 2026 — 6th Place Solution

Final Brier Score: "0.1181191"
Final Rank: 🥇 6th / 3,485 teams

---

📌 Overview

This repository contains my solution for the March Machine Learning Mania 2026 Kaggle competition.

The objective is to predict the probability that one NCAA team beats another, evaluated using Brier Score (mean squared error of probabilities).

My approach relies on a simple but highly effective ensemble of three independent modeling pipelines.

---

🧠 Solution Summary

The final submission is obtained by averaging predictions from three diverse models:

1️⃣ Elo-Based Model

- Margin-of-victory adjusted Elo ratings
- Seasonal regression (decay between seasons)
- Captures team strength dynamics

2️⃣ Four Factors Model

- Based on Dean Oliver's framework:
  - Effective FG% (eFG%)
  - Turnover Rate (TOV%)
  - Offensive Rebound % (ORB%)
  - Free Throw Rate (FTR)
- Represents true on-court performance

3️⃣ Context Model

- Late-season momentum
- Conference strength (via Elo aggregation)
- Coaching tournament experience

👉 Each pipeline is trained independently with different feature sets.

---

🔗 Final Ensemble

Predictions are combined using simple averaging:

final_pred = (pred_elo + pred_ff + pred_ctx) / 3

🔧 Post-processing
Applied minor calibration adjustments on select matchups (mainly women's tournament)
Integrated external signal from 25 human brackets:

P_final = 0.95 * P_model + 0.05 * P_human 
Minor manual corrections were applied during post-processing to fix input inconsistencies (e.g., accidental replacement of 0 with 2 due to keyboard input).

Clipped probabilities to [0.005, 0.995]
---

⚙️ Key Design Choices

- Model diversity over complexity
- Feature engineering > hyperparameter tuning
- Avoided heavy stacking to reduce overfitting
- Focused on stable, well-calibrated probabilities

---

📊 Validation

- Training data: historical NCAA tournaments
- Validation seasons: 2022–2025
- Metric: Brier Score

The ensemble consistently outperformed individual models in validation.

---

📉 Failure Analysis (Short)

- Struggled in:
  - Close matchups (similar seeds)
  - High-variance upset games
- Less sensitivity to:
  - Injury/news-based changes
  - Late unexpected form shifts

---

🛠️ Reproduction

1. Install Requirements

pip install -r requirements.txt

Or manually:

pip install numpy pandas scikit-learn lightgbm xgboost catboost kaggle

---

2. Download Data

You need Kaggle API access:

pip install kaggle
kaggle competitions download -c march-machine-learning-mania-2026 -p data/
cd data
unzip "*.zip"
rm *.zip
cd ..

---

3. Run Pipelines

Run the three notebooks (or scripts):

notebooks/
├── elo_model.ipynb
├── four_factors_model.ipynb
├── context_model.ipynb

Each notebook generates predictions.

---

4. Ensemble Predictions

final_pred = (pred1 + pred2 + pred3) / 3

Apply clipping:

final_pred = np.clip(final_pred, 0.05, 0.95)

---

📁 Repository Structure

.
├── data/
├── notebooks/
├── outputs/
├── submission/
├── SOLUTION_WRITEUP.md
└── README.md

---

💡 Key Takeaways

- Simple averaging of strong models can outperform complex ensembles
- Feature quality matters more than model complexity
- Calibration is critical for Brier Score optimization
- Diversity between models is the main driver of performance

---

🔮 Possible Improvements

- Weighted averaging instead of uniform
- Probability calibration (Isotonic / Platt scaling)
- External ranking systems (Massey Ordinals, KenPom, etc.)
- Separate models for men and women

---

🙌 Acknowledgments

Thanks to the Kaggle community for valuable discussions and shared insights.

---
