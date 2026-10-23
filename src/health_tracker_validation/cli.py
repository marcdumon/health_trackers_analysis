"""The command line: `htv collect` copies the new data of all sources, `htv build` makes the tables."""

import argparse
import json

from health_tracker_validation import build, collect


def main() -> None:
    """Run one command."""
    parser = argparse.ArgumentParser(prog='htv', description='Collect and prepare the data of the health tracker validation.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('collect', help='copy the new data of all sources into data/raw/')
    commands.add_parser('build', help='make the tables in data/processed/ from data/raw/')
    commands.add_parser('update', help='collect, then build')
    args = parser.parse_args()
    if args.command in ('collect', 'update'):
        print(json.dumps({'collect': collect.run()}, indent=1, default=str))
    if args.command in ('build', 'update'):
        print(json.dumps({'build': build.run()}, indent=1))
