from ortools.algorithms.python import knapsack_solver
import os
import csv
import time
groups = ['00Uncorrelated', '01WeaklyCorrelated', '02StronglyCorrelated', '03InverseStronglyCorrelated', '04AlmostStronglyCorrelated', '05SubsetSum', '06UncorrelatedWithSimilarWeights', '07SpannerUncorrelated',
          '08SpannerWeaklyCorrelated', '09SpannerStronglyCorrelated', '10MultipleStronglyCorrelated','11ProfitCeiling', '12Circle']
ranges = ['R01000', 'R10000']
sizes = ['n00050', 'n00100', 'n00200', 'n00500', 'n01000']
files = {'R01000':['s001.kp','s000.kp'], 'R10000':['s000.kp']}
with open('stats.csv', 'w') as output:
  writer = csv.writer(output)
  writer.writerow(['Group', 'Size', 'Range', 'File', 'Total Value', 'Total Weight', 'Time', 'Optimal?'])
  for group in groups:
    for r in ranges:
      for file in files[r]:
        for size in sizes:
          path = os.path.join(os.getcwd(), 'kplib', group, size, r, file)
          values = []
          weights = [[]]
          capacities = []
          with open (path, 'r') as f:
            inp = f.read().splitlines()

          capacities.append (int(inp[2]))

          for i in range(4, len(inp)):
            data = inp[i].split()
            values.append(int(data[0]))
            weights[0].append(int(data[1]))

          solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            "KnapsackExample",
        )

          solver.init(values, weights, capacities)
          solver.set_time_limit(180)

          start =  time.time()
          computed_value = solver.solve()
          end = time.time()

          packed_items = []
          packed_weights = []
          total_weight = 0

          for i in range(len(values)):
              if solver.best_solution_contains(i):
                  packed_items.append(i)
                  packed_weights.append(weights[0][i])
                  total_weight += weights[0][i]
          optimal = "YES" if solver.is_solution_optimal() else "NO"
          writer.writerow([group, size, r, file, computed_value, total_weight, '%.7f ' %(end-start), optimal])
