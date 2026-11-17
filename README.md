\# AI-Based Typosquatting Domain Detector



An AI-powered web application that detects suspicious and potentially typosquatting domains using machine learning and domain analysis.



\## Project Overview



Typosquatting is a cyberattack in which attackers register domains similar to legitimate websites to trick users. This project analyzes domain names and identifies potentially suspicious domains.



\## Features



\- Detects potentially malicious and typosquatting domains.

\- Uses machine learning for domain classification.

\- Analyzes domain similarity and lexical characteristics.

\- Uses trusted brand and domain information.

\- Provides a web interface using Streamlit.

\- Provides API functionality using Flask.



\## Technology Stack



\- Python

\- Machine Learning

\- XGBoost

\- Scikit-learn

\- NLP

\- Flask

\- Streamlit

\- Pandas and NumPy



\## Project Structure



```text

domain-detector/

├── app.py

├── api.py

├── requirements.txt

├── generate\_cm.py

├── dataset/

│   └── model/

│       ├── domain\_detector\_xgb.pkl

│       ├── hybrid\_meta.json

│       ├── hybrid\_scaler.pkl

│       ├── trusted\_brands.pkl

│       ├── trusted\_domains.pkl

│       └── typosquat\_hybrid.keras

├── scripts/

│   ├── check\_domain.py

│   ├── domain\_utils.py

│   ├── feature\_extraction.py

│   ├── nlp\_features.py

│   └── training and evaluation scripts

└── results/

```



\## Installation



1\. Clone this repository:



&#x20;  `git clone https://github.com/mansii-hub/REPOSITORY-NAME.git`



2\. Open the project folder.



3\. Install dependencies:



&#x20;  `pip install -r requirements.txt`



4\. Run the Streamlit application:



&#x20;  `streamlit run app.py`



\## Team



\- Mansi Gusain

\- Lakshita Chauhan

\- Saloni Negi



\## Academic Project



Developed as a B.Tech Computer Science and Engineering project at Graphic Era Hill University.

