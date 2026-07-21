import argparse
from datasets import load_dataset
from pipeline.orchestrator import run_pipeline

def main():
    
    parser = argparse.ArgumentParser(
        description="LegalMind — ECHR Multi-Agent Analysis System"
    )
    
    parser.add_argument(
        "--example",
        type=int,
        help="Uruchom przykład z datasetu (np. --example 1)"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        help="Własny tekst sprawy do analizy"
    )
    
    args = parser.parse_args()
    
    if args.example is not None:
        # Załaduj przykład z datasetu
        print(f"Loading example {args.example} from ECHR dataset...")
        ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
        case = ds['test'][args.example]
        run_pipeline(case['text'])
    
    elif args.text:
        # Własny tekst — podziel na paragrafy
        paragraphs = args.text.split(". ")
        run_pipeline(paragraphs)
    
    else:
        # Domyślnie uruchom przykład 0
        print("No arguments provided. Running default example...")
        ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
        case = ds['test'][0]
        run_pipeline(case['text'])


if __name__ == "__main__":
    main()