#!/bin/bash

# 定义 domains 数组
domains=("koran" "it" "law" "medical")

# 定义 vdb_types 数组
vdb_types=("base" "chat")

# 循环遍历每个 domain 和 vdb_type
for domain in "${domains[@]}"; do
    for vdb_type in "${vdb_types[@]}"; do
        echo "Processing domain: $domain with vdb_type: $vdb_type"
        
        # 创建索引
        python main.py create_index --domain "$domain" --vdb_type "$vdb_type"
        
    done
done

# 分析结果
python main.py analyze_results

echo "All processing completed."