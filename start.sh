#!/usr/bin/env bash
python3 opt.py --centrality betweenness --channels 5 --degree 2 --plot yes --poor-nodes yes
python3 -m http.server 5000
