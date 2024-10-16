import subprocess
import os
import shutil
import json

#BIEN PENSER A VIDER BENCHMARK AVANT DE RELANCER

def benchmark_one(player1, player2):
    command = [
    'python', 'main_divercite.py',
    '-t', 'local',
    player1, player2,
    '-r', '-g']
    subprocess.run(command)

def benchmark(player1, player2, max):
    wins = {player1[:-3] + "_1": 0, player2[:-3] + "_2" : 0, "draw" : 0}  # Initialize win counts
    for _ in range(max):
        benchmark_one(player1, player2)  # Assumed to run a match and generate a .json file
        for file in os.listdir():
            if file.endswith(".json"):
                # Read the contents of the JSON file
                with open(file, 'r') as f:
                    game_data = json.load(f)
                
                last_turn = game_data[-1]
                scores = last_turn["scores"]
                player_1 = last_turn["players"][0]
                player_2 = last_turn["players"][1]

                player_1_name = player_1["name"]
                player_2_name = player_2["name"]
                player_1_score = scores[str(player_1["id"])]
                player_2_score = scores[str(player_2["id"])]

                # Determine the winner
                if player_1_score > player_2_score:
                    winner = player_1_name
                elif player_2_score > player_1_score:
                    winner = player_2_name
                else:
                    winner = "draw"
                # Update wins count
                wins[winner] += 1
                
                # Move the JSON file to the "benchmark" directory
                if not os.path.exists("benchmark"):
                    os.makedirs("benchmark")  # Create the directory if it doesn't exist
                shutil.move(file, "benchmark")
                
    return wins

if __name__ == '__main__':
    os.makedirs("benchmark", exist_ok=True)
    wins = benchmark("random_player_divercite.py", "random_player_divercite.py", 10)
    with open("wins.txt", "w") as file:
        print(wins, file=file) 