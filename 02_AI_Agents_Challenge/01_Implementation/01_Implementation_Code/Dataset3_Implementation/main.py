"""
main.py - Entry point for the MirrorPay fraud detection agent.

Usage examples:
  # Quick test on small dataset with cheap model:
  python main.py --dataset "../../datasets/The Truman Show - train" --model cheap

  # Production run on validation dataset with heavy model:
  python main.py --dataset "../../datasets/The Truman Show - validation" --model heavy --output output_truman.txt

  # Use a specific OpenRouter model ID:
  python main.py --dataset "../../datasets/Brave New World - train" --model google/gemini-2.0-flash-001

  # Quiet mode (no step-by-step output):
  python main.py --dataset "../../datasets/Deus Ex - train" --model mid --quiet
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from agent import (
    MODELS,
    create_fraud_agent,
    flush_langfuse,
    generate_session_id,
    resolve_model_id,
    run_agent,
)
from tools import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MirrorPay Fraud Detection Agent – Reply Code Challenge 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Model presets:
  cheap  = {MODELS['cheap']}  (fast, free for testing)
  mid    = {MODELS['mid']}  (balanced)
  heavy  = {MODELS['heavy']}  (best quality, use for final eval)

Examples:
  python main.py --dataset "../../datasets/The Truman Show - train" --model cheap
  python main.py --dataset "../../datasets/Brave New World - validation" --model heavy -o output.txt
""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory (must contain transactions.csv, users.json, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cheap",
        help=f"Model preset ({', '.join(MODELS.keys())}) or a full OpenRouter model ID.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Model temperature. Lower = more deterministic. (default: 0.1)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path. Default: ./output_<dataset_name>.txt",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step agent output.",
    )

    return parser.parse_args()


def validate_env() -> None:
    """Check that all required env vars are set."""
    required = [
        "OPENROUTER_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "TEAM_NAME",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Fix: copy .env.example to .env in the repo root and fill in your values.")
        sys.exit(1)


def main() -> None:
    load_dotenv(find_dotenv())
    args = parse_args()

    # Validate environment
    validate_env()

    # Resolve dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        sys.exit(1)
    if not (dataset_path / "transactions.csv").exists():
        print(f"ERROR: transactions.csv not found in {dataset_path}")
        sys.exit(1)

    # Resolve model
    model_id = resolve_model_id(args.model)

    # Generate session ID
    session_id = generate_session_id()

    # Dataset name for display
    dataset_name = dataset_path.name

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
        output_path = Path(f"output_{safe_name}.txt")

    # --- Banner ---
    print("=" * 64)
    print("  MirrorPay Fraud Detection Agent")
    print("  Reply Code Challenge 2026 – The Eye")
    print("=" * 64)
    print(f"  Dataset:    {dataset_name}")
    print(f"  Path:       {dataset_path.resolve()}")
    print(f"  Model:      {model_id}")
    print(f"  Temp:       {args.temperature}")
    print(f"  Output:     {output_path}")
    print(f"  Session ID: {session_id}")
    print("=" * 64)

    # --- Load dataset ---
    print("\nLoading dataset...")
    meta = load_dataset(dataset_path)
    print(f"  Citizens:    {meta['citizens']}")
    print(f"  Transactions: {meta['transactions']}")
    print(f"  GPS pings:   {meta['location_pings']}")
    print(f"  SMS:         {meta['sms']}")
    print(f"  Emails:      {meta['mails']}")

    # --- Create and run agent ---
    agent = create_fraud_agent(model_id, args.temperature)
    flagged_ids: list[str] = []
    run_error: Exception | None = None
    try:
        flagged_ids = run_agent(
            agent,
            session_id=session_id,
            dataset_name=dataset_name,
            verbose=not args.quiet,
            primary_model_id=model_id,
            temperature=args.temperature,
        )
    except KeyboardInterrupt:
        print("\nERROR: Run interrupted by user. Output not finalized.")
        sys.exit(130)
    except Exception as exc:
        run_error = exc
        print(f"ERROR: Agent run failed: {exc}")
    finally:
        try:
            flush_langfuse()
        except KeyboardInterrupt:
            print("WARNING: Langfuse flush interrupted; exiting without waiting further.")

    if run_error is not None:
        sys.exit(1)

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for tid in flagged_ids:
            f.write(f"{tid}\n")

    if not flagged_ids:
        print("WARNING: Output is empty. This will be rejected by the challenge platform.")
    if len(flagged_ids) == meta["transactions"]:
        print("WARNING: All transactions were flagged. This will be rejected by the challenge platform.")

    # --- Summary ---
    print("\n" + "=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  Flagged transactions: {len(flagged_ids)} / {meta['transactions']}")
    print(f"  Output file:          {output_path.resolve()}")
    print(f"  Session ID:           {session_id}")
    print()
    print("  NEXT STEPS:")
    print("  1. Review the output file.")
    print("  2. Upload the .txt file on the challenge platform.")
    print(f"  3. Paste this Session ID: {session_id}")
    print("  4. For eval submissions: also upload your source code .zip")
    print("=" * 64)


if __name__ == "__main__":
    main()
