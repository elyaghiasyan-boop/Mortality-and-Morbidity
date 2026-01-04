"""
Advanced Maternal Mortality Simulation Framework
Using OOP, Random Distributions, Pandas, Numpy, CSV/JSON/SQLite, Threading, Logging
"""

# -----------------------------
# 1. Imports
# -----------------------------
import os, shutil, re, string, threading, time, subprocess, json, sqlite3, logging
import random, statistics, datetime
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict, OrderedDict, deque, namedtuple, ChainMap
from functools import reduce, lru_cache, partial, wraps
import requests
from itertools import chain, combinations

# -----------------------------
# 2. Logging Setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# -----------------------------
# 3. OOP Classes
# -----------------------------
class CountryMMR:
    """Country-level maternal mortality statistics"""
    def __init__(self, name, mmr, region):
        self.name = name
        self.mmr = mmr
        self.region = region
    
    def __repr__(self):
        return f"{self.name} ({self.region}) - MMR: {self.mmr}"
    
    def risk_category(self):
        if self.mmr >= 500: return "Very High"
        elif 300 <= self.mmr < 500: return "High"
        elif 100 <= self.mmr < 300: return "Moderate"
        return "Low"

class RegionMMR:
    """Aggregates countries into regions"""
    def __init__(self, name):
        self.name = name
        self.countries = []
    
    def add_country(self, country: CountryMMR):
        self.countries.append(country)
    
    def average_mmr(self):
        return statistics.mean([c.mmr for c in self.countries])

class GlobalMMR:
    """Global maternal mortality aggregation"""
    def __init__(self):
        self.regions = {}
    
    def add_region(self, region: RegionMMR):
        self.regions[region.name] = region
    
    def global_average(self):
        all_mmrs = [c.mmr for r in self.regions.values() for c in r.countries]
        return statistics.mean(all_mmrs)

# -----------------------------
# 4. Load / Prepare Data
# -----------------------------
data_file = Path("maternal_mortality.csv")
if not data_file.exists():
    df = pd.DataFrame([
        ["Nigeria", 993, "Sub-Saharan Africa"],
        ["Chad", 748, "Sub-Saharan Africa"],
        ["CAR", 692, "Sub-Saharan Africa"],
        ["South Sudan", 692, "Sub-Saharan Africa"],
        ["Liberia", 628, "Sub-Saharan Africa"],
        ["Somalia", 563, "Sub-Saharan Africa"],
        ["Afghanistan", 521, "Southern Asia"],
        ["Benin", 505, "Sub-Saharan Africa"],
        ["Guinea-Bissau", 518, "Sub-Saharan Africa"],
        ["USA", 21, "North America"],
        ["Iceland", 2, "Europe"],
        ["Mexico", 44, "Latin America"],
        ["Colombia", 59, "Latin America"]
    ], columns=["Country", "MMR", "Region"])
    df.to_csv(data_file, index=False)
else:
    df = pd.read_csv(data_file)

# -----------------------------
# 5. Initialize Objects
# -----------------------------
global_mmr = GlobalMMR()
regions_dict = {}

for idx, row in df.iterrows():
    country = CountryMMR(row["Country"], row["MMR"], row["Region"])
    if row["Region"] not in regions_dict:
        regions_dict[row["Region"]] = RegionMMR(row["Region"])
    regions_dict[row["Region"]].add_country(country)

for region in regions_dict.values():
    global_mmr.add_region(region)

logging.info(f"Global average MMR: {global_mmr.global_average():.2f}")

# -----------------------------
# 6. Random Simulations (ALL distributions)
# -----------------------------
simulation_results = {}

for region in global_mmr.regions.values():
    for country in region.countries:
        # Random distributions
        simulation_results[country.name] = {
            "Uniform": random.uniform(0.9*country.mmr, 1.1*country.mmr),
            "Gaussian": random.gauss(country.mmr, 0.1*country.mmr),
            "Normal": random.normalvariate(country.mmr, 0.1*country.mmr),
            "Exponential": random.expovariate(1/(country.mmr+1e-5)),
            "Beta": random.betavariate(2,5)*country.mmr,
            "Gamma": random.gammavariate(2, country.mmr/10),
            "Log-Normal": random.lognormvariate(np.log(country.mmr+1e-5), 0.1),
            "Pareto": (random.paretovariate(3)-1)*(country.mmr/2),
            "Von Mises": (random.vonmisesvariate(0,1)+3.14)*(country.mmr/10),
            "Original MMR": country.mmr,
            "Risk Category": country.risk_category()
        }

sim_df = pd.DataFrame(simulation_results).T
print("\n===== Simulated Maternal Mortality =====")
print(sim_df)

# -----------------------------
# 7. Threading Example: Simulate multiple runs
# -----------------------------
def simulate_multiple_runs(country_name, mmr):
    results = []
    for _ in range(5):
        results.append(random.gauss(mmr, 0.1*mmr))
    logging.info(f"{country_name} simulation runs: {results}")
    return results

threads = []
for c in df["Country"]:
    t = threading.Thread(target=simulate_multiple_runs, args=(c, df[df["Country"]==c]["MMR"].values[0]))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# -----------------------------
# 8. Save to CSV / JSON
# -----------------------------
sim_df.to_csv("simulated_mmr.csv")
sim_df.to_json("simulated_mmr.json", orient="records")

# -----------------------------
# 9. SQLite Storage
# -----------------------------
conn = sqlite3.connect("maternal_mortality.db")
sim_df.to_sql("simulated_mmr", conn, if_exists="replace", index=False)
conn.close()

# -----------------------------
# 10. Summary Statistics
# -----------------------------
all_values = sim_df.drop(columns=["Original MMR","Risk Category"]).values.flatten()
logging.info(f"Simulation Mean: {np.mean(all_values):.2f}, Median: {np.median(all_values):.2f}, Std: {np.std(all_values):.2f}")

# -----------------------------
# 11. Pandas Grouping by Risk Category
# -----------------------------
risk_group_stats = sim_df.groupby("Risk Category")[["Original MMR"]].agg(["mean","min","max"])
print("\n===== Risk Category Summary =====")
print(risk_group_stats)
