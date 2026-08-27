import json
import sys
from ctypes import util
import ctypes
from crec.observers import Screen
from crec import crec
import asyncio
import argparse
from dotenv.main import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description='A Python package with command-line interface')
    parser.add_argument('--user-name', '-u', type=str, default="anonymous", help='The user name to use')
    parser.add_argument('--check-permissions', action='store_true', help='Check input monitoring permissions and exit')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')

    # Scroll filtering options
    parser.add_argument('--scroll-debounce', type=float, default=0.5,
                        help='Minimum time between scroll events (seconds, default: 0.5)')
    parser.add_argument('--scroll-min-distance', type=float, default=5.0,
                        help='Minimum scroll distance to log (pixels, default: 5.0)')
    parser.add_argument('--scroll-max-frequency', type=int, default=10,
                        help='Maximum scroll events per second (default: 10)')
    parser.add_argument('--scroll-session-timeout', type=float, default=2.0,
                        help='Scroll session timeout (seconds, default: 2.0)')

    return parser.parse_args()


async def _main():
    args = parse_args()

    if args.check_permissions:
        try:
            # On macOS, we can check CoreGraphics
            app_services_path = util.find_library('ApplicationServices')
            result = False
            if app_services_path:
                core_graphics = ctypes.cdll.LoadLibrary(app_services_path)
                if hasattr(core_graphics, 'CGPreflightListenEventAccess'):
                    result = bool(core_graphics.CGPreflightListenEventAccess())

            print(json.dumps({"input_monitoring": result}))
            sys.exit(0)
        except Exception as e:
            print(json.dumps({"error": str(e), "input_monitoring": False}))
            sys.exit(1)

    print(f"User Name: {args.user_name}")

    # Create Screen observer with scroll filtering configuration
    screen_observer = Screen(
        debug=args.debug,
        scroll_debounce_sec=args.scroll_debounce,
        scroll_min_distance=args.scroll_min_distance,
        scroll_max_frequency=args.scroll_max_frequency,
        scroll_session_timeout=args.scroll_session_timeout,
    )

    async with crec(args.user_name, screen_observer):
        await asyncio.Future()  # run forever (Ctrl-C to stop)


def main():
    asyncio.run(_main())


if __name__ == '__main__':
    main()
