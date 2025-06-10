import gradio as gr
import pandas as pd
import numpy as np
import joblib
import random
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Load model and data (LOCAL paths)
model_path = 'Models/logistic_regression_model.pkl'
data_path = 'Datasets/testing_data.csv'

model = joblib.load(model_path)
data = pd.read_csv(data_path)
data['Player'] = ['Player' + str(i) for i in range(1, len(data) + 1)]

# Predict Skill Rating
X = data.drop(columns=['Target', 'Player'])
data['SkillRating'] = model.predict_proba(X)[:, 1]

# Assign rank and points to next rank
def assign_rank(score):
    if score < 0.4:
        return "Bronze", round((0.4 - score) * 100)
    elif score < 0.6:
        return "Silver", round((0.6 - score) * 100)
    elif score < 0.75:
        return "Gold", round((0.75 - score) * 100)
    elif score < 0.90:
        return "Platinum", round((0.90 - score) * 100)
    elif score < 0.95:
        return "Diamond", round((0.95 - score) * 100)
    elif score < 0.98:
        return "Master", round((0.98 - score) * 100)
    else:
        return "Grandmaster", round((1.00 - score) * 100)

data[['Rank', 'ToNextRank']] = data['SkillRating'].apply(lambda s: pd.Series(assign_rank(s)))
data['ToNextRank'] = data['ToNextRank'].astype(str) + ' pts'

# Player142 profile HTML block
player142_data = data[data['Player'] == 'Player142'].iloc[0]
player_profile = f"""
<div style='background-color:#2d3436; padding:20px; border-radius:10px; color:white; font-family:monospace'>
    <h2>Welcome, Player142</h2>
    <p><strong>Skill Rating:</strong> {player142_data['SkillRating']:.2f}</p>
    <p><strong>Rank:</strong> {player142_data['Rank']}</p>
    <p><strong>Points to next rank:</strong> {player142_data['ToNextRank']}</p>
    <p><strong>Preferred Mode:</strong> 5v5 Competitive</p>
    <p><strong>Current Form:</strong> Active</p>
</div>
"""

# Matchmaking logic
def simulate_matchmaking(mode, teammates):
    df_sorted = data.sort_values(by='SkillRating', ascending=False).reset_index(drop=True)
    player142 = df_sorted[df_sorted['Player'] == "Player142"].iloc[0]

    def generate_combined_avg_skill_chart(df1, df2):
        ranks = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Master", "Grandmaster"]
        avg1 = df1.groupby("Rank")["Skill Rating"].mean().reindex(ranks, fill_value=None)
        avg2 = df2.groupby("Rank")["Skill Rating"].mean().reindex(ranks, fill_value=None)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x=avg1.index, y=avg1.values, label="Team 1", marker="o", ax=ax)
        sns.lineplot(x=avg2.index, y=avg2.values, label="Team 2", marker="o", ax=ax)
        ax.set_title("Average Skill Rating by Rank (Team 1 vs Team 2)")
        ax.set_ylabel("Skill Rating")
        ax.set_ylim(0, 1.0)
        plt.xticks(rotation=15)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png'); buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    def generate_combined_skill_dist_chart(df1, df2):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(df1['Skill Rating'], linestyle="--", linewidth=2, label="Team 1", ax=ax)
        sns.kdeplot(df2['Skill Rating'], linestyle=":", linewidth=2, label="Team 2", ax=ax)
        ax.set_title("Skill Rating Distribution Comparison")
        ax.set_xlabel("Skill Rating")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png'); buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    if mode == "Solo Queue":
        solo = [player142['Player'], player142['SkillRating'], player142['Rank'], player142['ToNextRank']]
        pool = df_sorted[df_sorted['Player'] != "Player142"][['Player', 'SkillRating', 'Rank', 'ToNextRank']].values.tolist()
        for _ in range(100):
            random.shuffle(pool)
            team1 = [solo] + pool[:4]
            team2 = pool[4:9]
            fair = abs(sum(p[1] for p in team1) - sum(p[1] for p in team2))
            prec = round(1 / (1 + fair), 2)
            if prec > 0.8:
                df1 = pd.DataFrame(team1, columns=["Player", "Skill Rating", "Rank", "To Next Rank"])
                df2 = pd.DataFrame(team2, columns=["Player", "Skill Rating", "Rank", "To Next Rank"])
                avg_img = generate_combined_avg_skill_chart(df1, df2)
                dist_img = generate_combined_skill_dist_chart(df1, df2)
                return df1, df2, f"Precision Score: {prec} (Highly Balanced)", avg_img, dist_img
        return "No balanced match found", None, None, None, None

    elif mode == "Fixed Team":
        if not teammates or not (1 <= len(teammates) <= 4):
            return "Select 1 to 4 teammates", None, None, None, None
        team1 = [player142[['Player', 'SkillRating', 'Rank', 'ToNextRank']].tolist()]
        for p in teammates:
            row = df_sorted[df_sorted['Player'] == p].iloc[0]
            team1.append([row['Player'], row['SkillRating'], row['Rank'], row['ToNextRank']])
        needed = 5 - len(team1)
        ids = [p[0] for p in team1]
        pool = df_sorted[~df_sorted['Player'].isin(ids)][['Player', 'SkillRating', 'Rank', 'ToNextRank']].values.tolist()
        for _ in range(100):
            random.shuffle(pool)
            extra = pool[:needed]
            team2 = pool[needed:needed+5]
            team1_full = team1 + extra
            fair = abs(sum(p[1] for p in team1_full) - sum(p[1] for p in team2))
            prec = round(1 / (1 + fair), 2)
            if prec > 0.8:
                df1 = pd.DataFrame(team1_full, columns=["Player", "Skill Rating", "Rank", "To Next Rank"])
                df2 = pd.DataFrame(team2, columns=["Player", "Skill Rating", "Rank", "To Next Rank"])
                avg_img = generate_combined_avg_skill_chart(df1, df2)
                dist_img = generate_combined_skill_dist_chart(df1, df2)
                return df1, df2, f"Precision Score: {prec} (Highly Balanced)", avg_img, dist_img
        return "No balanced match found", None, None, None, None

# Player list excluding Player142
player_list = [p for p in data['Player'].unique() if p != 'Player142']

# Gradio UI
with gr.Blocks(css="""
.gradio-container {background-color: #1e1e2e; color: #f1f1f1; font-family: monospace;}
.gr-button {background: #74b9ff; color: black; font-weight: bold;}
input[type='checkbox']:checked {accent-color: #55efc4;}
""") as demo:

    with gr.Group(visible=True) as landing:
        gr.Markdown("# SBMM Matchmaking Prototype v3")
        gr.HTML(player_profile)
        start_btn = gr.Button("Start Simulation")

    with gr.Group(visible=False) as match_section:
        gr.Markdown("## Matchmaking Settings")
        gr.Markdown("*Choose a mode and simulate fair team formation.*")

        mode = gr.Radio(["Solo Queue", "Fixed Team"], label="Matchmaking Mode", value="Solo Queue")
        teammates = gr.CheckboxGroup(player_list, label="Select up to 4 Teammates", visible=False)
        mode.change(lambda m: gr.update(visible=(m == "Fixed Team")), inputs=mode, outputs=teammates)

        match_btn = gr.Button("Start Match")
        team1_out = gr.Dataframe(label="Team 1")
        team2_out = gr.Dataframe(label="Team 2")
        balance = gr.Text(label="Match Fairness")

        def start():
            return gr.update(visible=False), gr.update(visible=True)

        start_btn.click(fn=start, outputs=[landing, match_section])
        match_btn.click(
            fn=simulate_matchmaking,
            inputs=[mode, teammates],
            outputs=[
                team1_out, team2_out, balance,
                gr.Image(label="Average Skill Rating by Rank"),
                gr.Image(label="Skill Distribution (Team 1 vs Team 2)")
            ]
        )

demo.launch()
