#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_toy_all.sh
bash scripts/run_mnist_all.sh
bash scripts/run_cifar_smoke.sh
bash scripts/run_cifar_main.sh
