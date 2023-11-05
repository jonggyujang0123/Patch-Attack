#!/bin/bash

GID="$1"
EVENODD="$2"

sh scripts/run_Patch_MI.sh mnist "$GID" "$EVENODD"
sh scripts/run_Patch_MI.sh emnist "$GID" "$EVENODD"
sh scripts/run_Patch_MI.sh cifar10 "$GID" "$EVENODD"
