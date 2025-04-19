In Prototype v2, the matchmaking system is further enhanced to address different real-world scenarios of player grouping. It introduces two distinct scenarios: Fixed Team vs Matching Team and Solo Player Matchmaking. Both of these setups reflect common situations in online competitive games, where players may either play with a pre-existing team or solo. Let’s break it down:

### Scenario 1: Fixed Team vs Matching Team
In this scenario, Team 1 is fixed — meaning that the system starts with a pre-selected team of players. This could be the top 5 skilled players, a pre-made group of friends, or even a top-ranked team. The key idea is that this team is already in place and does not change.

The system then works to find a matching Team 2 by looking for players who can form a team that is balanced against the skill level of Team 1. To do this, the system randomly shuffles the remaining players and tests different combinations of players to form Team 2. The goal is to create two teams that are as skill-balanced as possible.

This scenario is very useful for real-world applications like clan matches or esports scrims, where a team of players already exists and the system must find a fair opponent team to play against.

### Scenario 2: Solo Player Matchmaking
In this scenario, Team 1 starts with one solo player. This represents a player who joins the game alone and needs to be placed into a balanced team. The system then attempts to form a team of 4 players around the solo player and match them against an opposing team of 5 players (Team 2).

The challenge here is to make sure that Team 1, with the solo player, is balanced against the full Team 2, which is composed of 5 players. This scenario mimics the ranked solo queue matchmaking seen in many competitive games, such as League of Legends, where individual players are placed into teams with others based on their skill ratings.
