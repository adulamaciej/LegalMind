from dotenv import load_dotenv

load_dotenv()

from evaluation.evaluation import run_evaluation

run_evaluation(n_cases=20, seed=42, judge_faithfulness=True)
