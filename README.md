# AI Network Intrusion Detection System

## Overview

This project implements a machine learning–based Network Intrusion Detection System (NIDS) that detects malicious traffic using behavioral patterns rather than static rule signatures.

The system is being developed incrementally with a strong focus on correctness, interpretability, and deployment readiness.

---

## Current System Capabilities

### 1. Baseline ML Foundation

A leak-free and structured machine learning pipeline was built using the NSL-KDD dataset:

- Structured data preprocessing and feature engineering  
- Binary classification setup (Normal vs Attack)  
- One-hot encoding of categorical traffic features  
- Feature scaling and stratified train–test split  
- Logistic Regression baseline model  
- Deployment-ready artifact saving (model, scaler, encoder)  

This phase established a reliable and reproducible ML foundation.

---

### 2. Model Strengthening & Architectural Refactoring

The detection engine was improved using a Random Forest ensemble model.

Key improvements:

- Significant reduction in false negatives  
- Non-linear modeling of traffic behavior  
- Feature importance analysis for interpretability  
- Validation that detection relies on meaningful security signals:
  - Traffic volume anomalies  
  - Error-rate spikes  
  - Service concentration patterns  

The codebase was refactored into modular components:

- `preprocessing.py`  
- `train.py`  
- `predict.py`  

This structure prepares the system for backend integration and real-world interaction.

---

## Design Philosophy

- Prioritize recall to minimize missed attacks  
- Validate model reasoning, not just metrics  
- Build modular, scalable architecture  
- Align machine learning behavior with real security principles  

---

## Next Phase

Backend integration, real-world interaction, and deployment-focused enhancements.
