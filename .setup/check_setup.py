"""
check_setup.py - Verifies that the environment is correctly configured.
Run via: make check
"""

import sys
import os

REQUIRED_VARS = [
    "OPENROUTER_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "TEAM_NAME",
]

REQUIRED_PACKAGES = [
    ("langchain", "langchain"),
    ("langchain_openai", "langchain-openai"),
    ("langgraph", "langgraph"),
    ("langfuse", "langfuse"),
    ("dotenv", "python-dotenv"),
    ("ulid", "ulid-py"),
    ("openai", "openai"),
    ("jupyter", "jupyter"),
    ("ipykernel", "ipykernel"),
]


def check_python():
    v = sys.version_info
    print(f"Python version: {v.major}.{v.minor}.{v.micro}")
    if v.major != 3 or not (10 <= v.minor <= 13):
        print(f"  ERROR: Python 3.10-3.13 required. Python {v.major}.{v.minor} is not supported.")
        print("         Python 3.14 is incompatible with Langfuse.")
        return False
    print(f"  OK: Python {v.major}.{v.minor} is supported.")
    return True


def check_packages():
    print("\nPackage imports:")
    all_ok = True
    for import_name, pkg_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
            print(f"  OK: {pkg_name}")
        except ImportError:
            print(f"  MISSING: {pkg_name}  ->  pip install {pkg_name}")
            all_ok = False
    return all_ok


def check_env():
    print("\nEnvironment variables (.env):")
    try:
        from dotenv import load_dotenv, find_dotenv
    except ImportError:
        print("  ERROR: python-dotenv not installed, cannot check .env")
        return False

    env_path = find_dotenv()
    if not env_path:
        print("  WARNING: No .env file found.")
        print("           Copy .env.example to .env in the repo root and fill in your credentials.")
        return False

    print(f"  Found .env at: {env_path}")
    load_dotenv(env_path)

    all_ok = True
    for var in REQUIRED_VARS:
        value = os.getenv(var, "")
        if not value or value.startswith("your-") or value.startswith("pk-your") or value.startswith("sk-your"):
            print(f"  NOT SET: {var}")
            all_ok = False
        else:
            # Mask sensitive values
            display = value if var in ("LANGFUSE_HOST", "TEAM_NAME") else value[:6] + "..." + value[-4:]
            print(f"  OK: {var} = {display}")

    return all_ok


def main():
    all_ok = True
    all_ok &= check_python()
    all_ok &= check_packages()
    all_ok &= check_env()

    print()
    if all_ok:
        print("All checks passed. You are ready to run the challenge.")
    else:
        print("Some checks failed. Fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
