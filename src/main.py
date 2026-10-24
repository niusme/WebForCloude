"""DeepSeek Web Chat automation — CLI entry point.

Usage:
    # One-time login (headed browser)
    python src/main.py --mode login

    # Single question (headless, reuses saved session)
    python src/main.py --mode chat --message "What is Python?"

    # Multi-turn demo
    python src/main.py --mode demo
"""

import argparse
import logging
import os
import sys

# Ensure the project root is on sys.path so "from src.xxx" works
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from src.deepseek import (
    DeepSeekClient,
    DeepSeekError,
    ResponseTimeoutError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _run_client(client: DeepSeekClient, fn, *args, **kwargs):
    """Run a function with the client, ensuring cleanup before any exit."""
    try:
        return fn(client, *args, **kwargs)
    except (DeepSeekError, ResponseTimeoutError) as exc:
        print(f"\nX Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    finally:
        print("Closing browser...")
        client.close()


def cmd_login(client: DeepSeekClient) -> int:
    """Open headed browser and let user log in manually."""
    client.start()
    client.login()
    return 0


def cmd_chat(client: DeepSeekClient, message: str) -> int:
    """Send a single message and print the response (headless)."""
    client.start()

    if not client.is_logged_in():
        print("X No valid session. Run --mode login first.")
        return 1

    print(f'Sending: "{message}"')
    response = client.ask(message)
    print(f"\n{'=' * 60}")
    print("  Response:")
    print(f"{'=' * 60}")
    print(response)
    print(f"{'=' * 60}")
    return 0


def cmd_demo(client: DeepSeekClient) -> int:
    """Run a multi-turn conversation demo."""
    print("=" * 60)
    print("  DeepSeek Multi-Turn Demo")
    print("=" * 60)

    client.start()

    if not client.is_logged_in():
        print("X No valid session. Run --mode login first.")
        return 1

    questions = [
        "Who are you? Please introduce yourself briefly.",
        "What can you help me with? List 3 things.",
        "Tell me a fun fact about space.",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n--- Turn {i} ---")
        print(f"  Q: {question}")
        print("  (waiting for response...)")
        try:
            response = client.ask(question)
            print(f"  A: {response[:300]}{'...' if len(response) > 300 else ''}")
        except (ResponseTimeoutError, DeepSeekError) as exc:
            print(f"  X Error on turn {i}: {exc}")
            break

    # Demonstrate starting a new conversation
    print(f"\n--- New Conversation ---")
    print("  Starting fresh conversation...")
    client.start_new_conversation()

    final_q = "Start fresh. What is 2+2?"
    print(f"  Q: {final_q}")
    try:
        response = client.ask(final_q)
        print(f"  A: {response}")
    except (ResponseTimeoutError, DeepSeekError) as exc:
        print(f"  X Error: {exc}")

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print(f"{'=' * 60}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSeek Web Chat automation via Playwright",
    )
    parser.add_argument(
        "--mode",
        choices=["login", "chat", "demo"],
        default="demo",
        help="Operation mode (default: demo)",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Hello!",
        help="Message to send (chat mode)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run with visible browser (default: headless)",
    )

    args = parser.parse_args()
    headless = not args.headed
    if args.mode == "login":
        # Manual login needs a visible window; force headed mode.
        if headless:
            print("Info: --mode login always runs with a visible browser window.")
        headless = False
    client = DeepSeekClient(headless=headless)

    if args.mode == "login":
        exit_code = _run_client(client, cmd_login)
    elif args.mode == "chat":
        exit_code = _run_client(client, cmd_chat, args.message)
    elif args.mode == "demo":
        exit_code = _run_client(client, cmd_demo)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
