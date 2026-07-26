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
        print(f"Loading example {args.example} from ECHR dataset...")
        ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
        if args.example < 0 or args.example >= len(ds['test']):
            print(f"Error: example index {args.example} is out of range (0-{len(ds['test'])-1})")
            return
        case = ds['test'][args.example]
        try:
            run_pipeline(case['text'])
        except Exception as e:
            print(f"Pipeline failed: {e}")

    elif args.text:
        paragraphs = [p.strip() for p in args.text.split("\n") if p.strip()]
        try:
            run_pipeline(paragraphs)
        except Exception as e:
            print(f"Pipeline failed: {e}")

    else:
        print("No arguments provided. Running default example...")
        ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
        case = ds['test'][0]
        try:
            run_pipeline(case['text'])
        except Exception as e:
            print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()