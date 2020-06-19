#!/bin/bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
gunicorn wsgi:app -b 0.0.0.0:5000 -t 1000 --workers 1 --threads 1
