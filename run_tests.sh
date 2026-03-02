#!/bin/bash
set -e

echo "START_TIME=$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"

for i in 1 2 3 4 5; do
  echo "=== Round $i ==="
  pytest tests-demo/test_demo_claude.py -v &
  pytest tests-demo/test_demo_deepseek.py -v &
  pytest tests-demo/test_demo_glm.py -v &
  pytest tests-demo/test_demo_minimax.py -v &
  pytest tests-demo/test_demo_openai.py -v &
  wait
done

echo "END_TIME=$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"
echo "All 5 rounds complete."
