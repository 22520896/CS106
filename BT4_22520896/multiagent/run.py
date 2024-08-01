import subprocess
import re
import csv
import time

layouts = ["capsuleClassic", "contestClassic", "mediumClassic", "minimaxClassic", "testClassic"]
agents = ["MinimaxAgent","AlphaBetaAgent", "ExpectimaxAgent"]
evalFns = ["scoreEvaluationFunction", "betterEvaluationFunction","betterEvaluationFunction2"]
ghosts = ["RandomGhost", "DirectionalGhost"]
seeds = ["22520896", "22520897", "22520898", "22520899", "22520900"]

results = {
    "RandomGhost": {evalFn: [] for evalFn in evalFns},
    "DirectionalGhost": {evalFn: [] for evalFn in evalFns}
}
with open('stats.csv', 'w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(['Ghost','Evaluation Fuction','Layout', 'Agent', 'Average Score', 'Win Count', 'Time'])
    for layout in layouts:
        for agent in agents:
            for ghost in ghosts:
                for evalFn in evalFns:
                    all_scores = []
                    win_count = 0
                    time_start = time.time()
                    for seed in seeds:
                        cmd = f"python pacman.py -l {layout} -p {agent} -a depth=3,evalFn={evalFn} -g {ghost} -s {seed} --frameTime 0"
                        print(f"Running: {cmd}")
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                        output, error = process.communicate()
                        if output:
                            output_str = output.decode('utf-8')
                            score_match = re.findall(r"Scores:\s+(-?[\d\.]+)", output_str)
                            win_match = re.findall(r"Record:\s+Win", output_str)
                            individual_scores = [float(score) for score in score_match]
                            all_scores.extend(individual_scores)
                            win_count += len(win_match)
                            print(f"Scores for seed {seed}: {individual_scores}")
                    time_end = time.time()
                    if all_scores:
                        average_score = sum(all_scores) / len(all_scores)
                    else:
                        average_score = 0
                    results[ghost][evalFn].append({
                        "layout": layout,
                        "agent": agent,
                        "individual_scores": all_scores, 
                        "average_score": average_score,
                        "win_count": win_count,
                        "time": time_end - time_start
                    })
                    writer.writerow([ghost, evalFn, layout, agent, average_score, f"{win_count}/5",'%.7f' % (time_end - time_start)])
                print("==============================================")


for ghost, evalFn_data in results.items():
        for evalFn, data in evalFn_data.items():
            print(f"Results for {ghost} with {evalFn}:")
            for result in data:
                print(f" - Layout: {result['layout']}, Agent: {result['agent']}, Scores: {result['individual_scores']}, Average Score: {result['average_score']}, Win Count: {result['win_count']}/5, Time: {result['time']}")
            print("\n")
