# Predicting Human DecisionMaking in Autonomous Vehicle Moral Dilemmas
**Team 12: Aileen Li, Alex Edwards, Carol Le, Mikyung Oh, Radha Pawar**

This project explores whether human moral judgments in autonomous-vehicle crash dilemmas can be predicted using machine learning. Using 100,000 outcome-level samples from MIT’s Moral Machine dataset, we model the probability that a given outcome (“stay” or “swerve”) will be chosen by a participant.
Our work includes exploratory data analysis, predictive modeling, interpretability via SHAP, and segmentation analysis to understand cross-cultural and scenario-based variation in moral preferences.

This repository contains all code and materials needed to reproduce our results.

## Project Overview

Autonomous vehicles may face unavoidable crash scenarios requiring moral decisions: Should the car stay on course and hit one group, or swerve and hit another?
The Moral Machine dataset captures millions of such decisions from people around the world.

Our goals:

+ Identify which scenario and character attributes most influence moral choices
+ Train ML models (CatBoost, XGBoost, Random Forest, Logistic Regression) to predict outcomes
+ Use SHAP values to understand how models weigh morally relevant features
+ Analyze segmentation across countries, scenario types, and latent clusters
+ Discuss implications for AI ethics and real-world automated decision systems

## Key Methods & Models

We evaluate multiple supervised learning models to predict decision outcomes:
+ CatBoost (best performing, AUC ≈ 0.78)
+ XGBoost
+ Random Forest
+ Logistic Regression

Interpretability:
+ SHAP (SHapley Additive exPlanations) to estimate each feature’s moral “weight”

Segmentation:
+ Performance breakdown by country, scenario type, and latent personas learned via K-Means clustering.

## Summary of Findings
+ People show consistent moral patterns across countries
+ Younger individuals, legal crossers, and larger groups were more often saved
+ Humans were prioritized over animals
+ Certain categories (e.g., social status dilemmas) were much harder to predict
+ CatBoost captured the strongest moral prediction signal

These findings provide insight into how AI ethical systems might align with human intuition and where human preferences are too inconsistent for reliable modeling.

## Dataset
Moral Machine Dataset (MIT Media Lab, via OSF):
https://osf.io/3hvt2/

We used a random subset of 100,000 rows from SharedResponse.csv for computational feasibility.
