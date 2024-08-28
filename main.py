"""A program with multi-stage processing and argument parsing."""

import argparse
import json
import os
import sys
import config
from loguru import logger


def parse_main_args():
    """Parse the main command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-stage processing program.")
    parser.add_argument(
        "stage",
        choices=["create_index", "process_data", "analyze_results"],
        help="Specify which stage to run",
    )
    return parser.parse_known_args()


def parse_create_index_args(args):
    """Parse arguments specific to the create_index stage."""
    parser = argparse.ArgumentParser(description="Create index stage.")
    parser.add_argument(
        "--domain", type=str, help="The field to be indexed", default="koran"
    )
    parser.add_argument(
        "--vdb_type",
        type=str,
        default="base",
        help="Define which LLM to be used as embedding model",
    )
    parser.add_argument("--prompt_type", type=str, help="Type of prompt to use")
    return parser.parse_args(args)


def create_index(args):

    from LLaMAEmbeddings import LLaMAEmbedding
    from faissManager import FAISS

    """Create index stage implementation."""
    print(f"Creating index for domain: {args.domain}")
    print(f"Using VDB type: {args.vdb_type}")
    print(f"Prompt type: {args.prompt_type}")

    raw_dataset_dir = f"{config.vdb_path}/{args.domain}/train.en"
    save_dir = f"{config.vdb_path}/{args.domain}/{config.vdb[args.vdb_type]}"

    if os.path.isfile(raw_dataset_dir):
        raw_data = list(open(raw_dataset_dir).readlines())

    embedding = LLaMAEmbedding(prompt=None, use_prompt=False, vdb_type=args.vdb_type)
    # 第一次索引找小集合
    raw_db = FAISS.build_knn_datastore(
        documents=raw_data,
        embedding=embedding,
        tensorSaveDir=None,
        vdb_type=args.vdb_type,
    )

    # print('raw_data',raw_data)
    raw_db.save_local(save_dir)
    print("knn数据库索引成功保存到了", save_dir)

    # Implementation here


def process_data():
    """Process data stage implementation."""
    print("Processing data...")
    # Implementation here


def analyze_results():
    """Analyze results stage implementation."""
    print("Analyzing results...")
    # Implementation here


def main():
    """Main function to run the appropriate stage based on user input."""
    args, unknown = parse_main_args()
    stage_functions = {
        "create_index": create_index,
        "process_data": process_data,
        "analyze_results": analyze_results,
    }

    stage_function = stage_functions.get(args.stage)
    if stage_function:
        if args.stage == "create_index":
            stage_args = parse_create_index_args(unknown)
            stage_function(stage_args)
        else:
            stage_function()
    else:
        print(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
