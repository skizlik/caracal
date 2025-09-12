#!/bin/bash
echo "finding actual NVIDIA CUDA tags..."
curl -s "https://registry.hub.docker.com/v2/repositories/nvidia/cuda/tags/?page_size=100" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for tag in data['results']:
    name = tag['name']
    if 'devel' in name and 'ubuntu' in name and ('11.' in name or '12.' in name):
        print(name)
" | head -10
