# 36120-25SP-AT3-solana-api

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

FastAPI for Solana next-day HIGH price prediction (AT3)
## 🚀 Overview

This project provides:
- A **FastAPI** backend for serving cryptocurrency analytics and model predictions.
- Integration with APIs such as Kraken, CoinGecko, TokenMetrics, and CoinDesk.
- Modular code for **data ingestion**, **feature extraction**, **model training**, and **visualization**.
- A production-ready structure following the **Cookiecutter Data Science** convention.

---

## 🗂️ Project Structure


```
├── LICENSE <- Open-source license if one is chosen
├── Makefile <- Makefile with convenience commands like make run, make test, etc.
├── README.md <- The top-level README for developers using this project
│
├── app 
│ ├── init.py
│ ├── main.py 
├── docs <- Project documentation and technical notes
│
├── models <- Serialized models and output predictions
│ ├──tuned_elasricnet_model.joblib
│  
├── notebooks <- Jupyter notebooks for exploration and experimentation
├── reports <- Generated analysis reports, figures, and summaries
├── requirements.txt <- Python dependencies for reproducing the environment
├── pyproject.toml <- Project metadata and configuration for tools like black
├── Dockerfile <- Docker configuration for containerized deployment
├── .gitignore <- Ignore patterns for git
└── tests
```

--------

