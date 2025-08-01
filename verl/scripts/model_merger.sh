#!/bin/bash
script_dir=$(cd "$(dirname "$0")" && pwd)

local_dir=xxx
target_dir=xxx
python  $script_dir/model_merger.py --local_dir $local_dir --backend fsdp --target_dir $target_dir --use_score