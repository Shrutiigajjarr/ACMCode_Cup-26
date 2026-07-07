# ⚽ TactixAI

AI-powered football tactical assistant and tournament pathway simulator.

Developed for **ACM Code Cup 2026 Prototype Round**.

---

# Team

- Shruti Gajjar (lead)
- Om Tailor
- Utssavi manjanwala
- aditya patil

---

# Problem Statement

Football teams often struggle to make quick tactical decisions before and during matches.

Current analysis tools are expensive, require expert analysts, or are difficult for smaller teams to use.

TactixAI provides an AI-assisted platform that helps coaches analyze teams, predict match outcomes, and simulate tournament progression.

---

# Features

### Tactical Support Assistant

- Select any two teams
- AI-based match outcome prediction
- Win / Draw / Loss probability
- Team strength comparison
- Formation visualization
- Player statistics dashboard

### Tournament Pathway Simulation

- Tournament progression visualization
- Knockout pathway simulation
- Bottleneck identification
- AI-generated tactical recommendations

---

# Tech Stack

## Frontend

- Streamlit

## Backend

- Python

## Machine Learning

- Scikit-learn
- Random Forest Classifier

## Data Processing

- Pandas
- NumPy

## Dataset

- Kaggle International Football Results Dataset

---

# Project Structure

```
ACM_HACKATHON/

│── app.py
│── requirements.txt
│── README.md
│
├── data/
│   ├── results.csv
│   └── players.csv
│
├── assets/
│
└── images/
```

---

# Installation

Clone the repository

```bash
git clone <repository-link>
```

Move into the project folder

```bash
cd ACM_HACKATHON
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

If Streamlit is not recognized:

```bash
python -m streamlit run app.py
```

---

# Dataset

This project uses the **International Football Results** dataset from Kaggle.

Files used:

- results.csv
- players.csv

---

# Future Improvements

- LightGBM based prediction model
- Live match statistics integration
- Player injury analysis
- Formation recommendation engine
- Interactive tournament bracket
- Performance analytics dashboard

---

# Demo

Prototype demonstration video:



---


---

# License

This project was developed for the ACM Code Cup 2026 Hackathon.
