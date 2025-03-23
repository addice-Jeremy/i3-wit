# 🔍 What-If Tool (WIT) — Exploring Movie Recommendations

This repository demonstrates the use of Google’s [What-If Tool (WIT)](https://pair-code.github.io/what-if-tool/) to inspect and interactively explore a movie recommendation model. The work was done as part of the **Machine Learning in Production (17-445/17-645, Spring 2025)** course at CMU.

## 📌 Context

In our class project, we built a recommendation system that suggests top movies for users based on their profile and viewing history. While the model achieved solid performance metrics, understanding *why* certain movies were recommended remained a challenge.

That’s where WIT came in.

## 🧪 What We Did

We tested the What-If Tool by feeding it user data and corresponding movie predictions. This allowed us to:

- Visually inspect how user attributes (like age or subscription type) affect model outputs
- Experiment with counterfactuals (e.g. “What if this user was younger?”)
- Identify biases or overly sensitive features
- Simulate different user scenarios without retraining the model

## 📁 Files Used

The following files were used to run WIT on the movie streaming scenario:

- `user_data_full.csv`: Contains user IDs and features like age, gender, and occupation
- `recommendation_data.csv`: Predicted top movies for each user
- 
We merged and sampled from these datasets to create inputs for WIT.

## 🧠 Example Usage

```python
import pandas as pd
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

# Load sample user input data
user_df = pd.read_csv("user_data_full.csv")
recs_df = pd.read_csv("recommendation_data.csv")
merged_df = pd.merge(user_df, recs_df, on="user_id")

sample_df = merged_df.sample(100)
wit_inputs = sample_df.to_dict(orient='records')

# Define a dummy prediction function for WIT
class DummyModel:
    def predict(self, examples):
        return [example["top_movie"] for example in examples]

model = DummyModel()

# Launch WIT
config = WitConfigBuilder(wit_inputs).set_custom_predict_fn(model.predict)
WitWidget(config)
