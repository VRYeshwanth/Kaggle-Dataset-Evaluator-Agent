from pprint import pprint
from analysis import load_dataset, run_full_analysis

df = load_dataset(r"data\heart_disease.csv")
report = run_full_analysis(df)

pprint(report)