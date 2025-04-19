In this prototype, we are simulating a **5v5 matchmaking system** for an online competitive game based on player skill ratings. The primary goal of this system is to ensure that the teams formed for a match are as balanced as possible, which helps create fairer and more enjoyable gameplay experiences for players.

The first step in the prototype is **loading the pre-trained logistic regression model**. This model has already been trained using data from players in a competitive game and saved in a `.pkl` file. By loading this model, we can use it to make predictions about player skill levels based on their past performance.

Next, we load the **testing dataset** that contains data on player performance in the game. This dataset is in CSV format and includes various features that describe a player's actions and performance, such as actions per minute (APM), hotkey usage, units made, and more. We also add a new column, **'Player'**, to uniquely identify each player by their index (e.g., Player1, Player2, etc.).

After preparing the data, the model is used to make **predictions about player skill levels**. The model assigns a skill rating to each player, which is represented as a probability (i.e., how confident the model is about a player's skill level). These predicted skill ratings are then used to **sort the players** in descending order, so that the most skilled players are ranked at the top.

To create the teams, the system **shuffles the players randomly** to ensure that each simulation results in different team compositions. The shuffled players are then divided into two teams, **Team 1** and **Team 2**, each containing 5 players. The goal is to balance the teams as much as possible based on their skill levels.

Once the teams are formed, we calculate the **total skill of each team** by summing the individual skill ratings of the players in the team. This gives us an overall skill score for both Team 1 and Team 2. We then calculate the **fairness of the teams** by finding the absolute difference in the total skill of both teams. If the skill difference is small, the teams are more balanced, and vice versa.

To evaluate how well the teams are balanced, we calculate the **precision**, which is a measure of how balanced the teams are. The precision is calculated using the formula `1 / (1 + fairness)`, so a lower skill difference results in a higher precision value. A precision score closer to 1 indicates that the teams are highly balanced, while a lower precision score means that the teams are more imbalanced. Based on the precision score, we classify the teams as either "Highly Balanced," "Moderately Balanced," or "Imbalanced."

Lastly, to provide a **visual representation of player skill levels**, we apply **Principal Component Analysis (PCA)**. PCA is a technique used to reduce the complexity of the data by converting the high-dimensional player data into two principal components (PCA1 and PCA2). This allows us to visualize the players in a 2D space. The players' skill ratings are used to color the points in the plot, giving us an intuitive view of how players are distributed based on their skill levels.

By combining these steps, the prototype simulates a matchmaking system that not only pairs players into teams but also ensures the teams are fairly balanced, using both numerical metrics (like total skill and precision) and visualizations (like PCA). This simulation provides insights into how well the system can match players of similar skill levels and helps in assessing whether the matchmaking system works as intended. 

The prototype is useful for testing and improving the **SBMM model** for online competitive games, allowing for fairer, more enjoyable matches based on skill levels.
