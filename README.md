# Data Science Bootcamp Projects

Shorter analysis projects from the Digital Skola Data Science Bootcamp (2026), rebuilt as clean, executed notebooks with business-focused findings.

## Projects in this repo

### 1. [Data Science Salaries: what pays and where?](data_science_salaries.ipynb)
pandas wrangling and EDA on 3,755 salary records (2020–2023).
* The median salary roughly **doubles from entry level (USD 70k) to senior (USD 146k)**.
* **Fully remote roles pay about the same as on-site roles** at the median (USD 135k vs 140k). Hybrid roles pay less.
* The share of fully remote roles fell from **53% in 2022 to 33% in 2023**.

![Salary overview](images/salary_overview.png)

### 2. [E-commerce sales and profit analysis](ecommerce_sales_analysis.ipynb)
KPI, category, region, shipping and discount analysis on 51,282 order lines (2023).
* **Fashion brings 65% of revenue** at the best margin (47.7%).
* **Discounts give away margin without adding volume.** Quantity stays at about 3 units per order at every discount level, while the margin drops from 50% to 41%.
* Revenue is steady through the year (the best month is only about 14% above the weakest).

![Revenue by category and top products](images/category_products.png)

## Featured projects (separate repos)

| Project | What it shows |
| --- | --- |
| [Flight Price Prediction](https://github.com/mirsyaddzulfaqor-tech/flight-price-prediction) | Regression (R² 0.985), booking-window insight, Streamlit app. Bootcamp final project |
| [Customer Churn Prediction](https://github.com/mirsyaddzulfaqor-tech/customer-churn-prediction) | Classification with imbalance handling; the top 20% of the ranking reaches 50% of churners |
| [Customer Segmentation](https://github.com/mirsyaddzulfaqor-tech/customer-segmentation) | K-Means clustering into 4 marketing personas |
| [Donor Reactivation System](https://github.com/mirsyaddzulfaqor-tech/donor-reactivation-demo) | RFM + ML priority queue, live Streamlit and Looker Studio demo |

## Run

`pip install -r requirements.txt`, then open any notebook. `python build_notebooks.py` re-executes both notebooks from scratch.

Data: public teaching datasets (`data/`).

---
Muhammad Irsyad Dzulfaqor · [LinkedIn](https://www.linkedin.com/in/mirsyaddzulfaqor/) · [GitHub](https://github.com/mirsyaddzulfaqor-tech)
