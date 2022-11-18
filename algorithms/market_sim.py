import random

holdings = 500
inc = 1.01
days = 300
min_prob = 80

avg_profit = holdings

num_simulations = 1

for i in range(num_simulations):
    for j in range(days):
        chance = random.randint(0, 100)
        if chance <= min_prob:
            holdings *= inc
        else:
            holdings / 2
    if holdings < 500:
        print("Simulation failed")

    avg_profit = (holdings + avg_profit) / 2
print("Average holdings: ", avg_profit)
