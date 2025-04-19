# Enhancing Skill-Based Matchmaking (SBMM) in Competitive Games Using Machine Learning

## Overview

This research focuses on enhancing **Skill-Based Matchmaking (SBMM)** in competitive online games. The goal is to create a system that pairs players based on their skill levels to ensure fairer and more balanced matches. This is achieved by using machine learning models to predict player skill ratings and utilizing those ratings for matchmaking.

The research specifically concentrates on **Real-Time Strategy (RTS)** games, with **StarCraft II** used as the primary test case. The study aims to improve the matchmaking experience by ensuring players with comparable skills are placed on opposite teams.

---

## Key Sections

### 1. **Dataset**

The data used in this research comes from the [StarCraft II Replay Analysis Dataset](https://www.kaggle.com/datasets/sfu-summit/starcraft-ii-replay-analysis), which includes various player performance metrics such as **APM (Actions Per Minute)**, **Hotkey usage**, and **game outcomes**. This dataset is used to train the machine learning models that predict player skill.

### 2. **Data Preprocessing**

Before training the models, the data went through several preprocessing steps:
- **Handling missing values**: Missing values were filled using the column's mean.
- **Feature Engineering**: A new feature called **Hotkey Efficiency** was created, which is the ratio of `SelectByHotkeys` to `AssignToHotkeys`.
- **Label Encoding**: The **LeagueIndex** was encoded into numerical values to serve as the target variable for classification.
- **Scaling & Normalization**: Features were scaled using **StandardScaler**, and vector normalization was applied to each row to prepare the dataset for PCA (Principal Component Analysis).

The final preprocessed dataset was then used for model training.

### 3. **Model Comparison**

Several machine learning models were trained and compared to predict player skill. These models included:
- **Random Forest Classifier**
- **Decision Tree Classifier**
- **Gradient Boosting Classifier**
- **Naive Bayes Classifier**

The models were evaluated based on accuracy, precision, recall, and F1-score, with the **Gradient Boosting Classifier** yielding the best performance.

#### [Model Comparison Notebook](https://github.com/LaXnZ/enhancing-sbmm-in-competitive-games/blob/model-comparison/Model_Comparison_with_Original_Dataset.ipynb)

### 4. **Model Training**

After preprocessing the data, I trained a **Logistic Regression** model, which is used to predict player skill levels (used for matchmaking).

#### [Model Training Notebook](https://github.com/LaXnZ/enhancing-sbmm-in-competitive-games/blob/model-training/Preprocessing_Dataset_%2B_Training_a_Model_Using_LogisticRegression.ipynb)

### 5. **Model Evaluation**

The trained model was evaluated using various performance metrics such as:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)
- **AUC (Area Under the Curve)**
- **ROC Curve**

This evaluation helped assess how well the model performed in predicting the skill ratings of players.

#### [Model Evaluation Folder](https://github.com/LaXnZ/enhancing-sbmm-in-competitive-games/tree/model-evaluation)

### 6. **Prototype v1 (Basic Matchmaking)**

**Prototype v1** focuses on random matchmaking. The system:
- Uses the predicted skill ratings to form two balanced 5v5 teams.
- Randomly shuffles players and matches them until balanced teams are found.
- Ensures that both teams have nearly equal total skill.

This method is used for general matchmaking where no player or team is fixed, and every player has an equal chance to be placed on either team.

#### [Prototype v1 Notebook](https://github.com/LaXnZ/enhancing-sbmm-in-competitive-games/blob/prototype-v1/SBMM_Prototype_V1_0.ipynb)

### 7. **Prototype v2 (Advanced Matchmaking Scenarios)**

**Prototype v2** extends the functionality with more complex matchmaking scenarios:
- **Scenario 1: Fixed Team vs Matching Team**: A pre-defined fixed team (e.g., a team of 5 friends) is matched against a randomly selected team with a similar total skill level.
- **Scenario 2: Solo Player Matchmaking**: A solo player is placed in a fixed team, and the rest of the players are shuffled to form balanced teams, similar to ranked solo-queue matchmaking.

#### [Prototype v2 Notebook](https://github.com/LaXnZ/enhancing-sbmm-in-competitive-games/blob/prototype-v2/SBMM_Prototype_V2_0.ipynb)

---

## Future Work

- **Model Extension**: I plan to extend the existing model to predict player ranks and forecast match outcomes (e.g., predicting who is likely to win or who is likely to rank up quickly).
- **Real-World Application**: I aim to explore integrating the matchmaking algorithm with real-world games or esports tournaments to test its effectiveness in a live environment.


## Conclusion

This project focuses on improving matchmaking in competitive online games, using machine learning to predict player skill levels and pair players accordingly. The goal is to enhance the gaming experience by ensuring fairer and more competitive matches, making the game enjoyable for players of all skill levels.

