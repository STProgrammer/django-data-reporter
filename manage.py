#!/usr/bin/env python
import os
import sys

def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'data_reporter.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Django isn't installed. Install dependencies from requirements.txt."
        ) from exc
    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
