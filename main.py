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
        choices=["create_index", "generate", "analyze_results"],
        help="Specify which stage to run",
    )
    # argument used by create index
    parser.add_argument(
        "--domain", type=str, help="The field to be indexed", default="koran"
    )
    parser.add_argument(
        "--vdb_type",
        type=str,
        default="base",
        help="Define which LLM to be used as embedding model",
    )
    return parser.parse_args()


def create_index(args):
    from LLaMAEmbeddings import LLaMAEmbedding
    from faissManager import FAISS
    
    """Create index stage implementation."""
    print(f"Creating index for domain: {args.domain}")
    print(f"Using VDB type: {args.vdb_type}")

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

    raw_db.save_local(save_dir)
    print("knn数据库索引成功保存到了", save_dir)


def nmt_with_lm(args):
    """Process data stage implementation."""
    print("Processing data...")
    print(f"Using domain: {args.domain}")
    print(f"Using VDB type: {args.vdb_type}")

    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
    from knn_lm import KNNWrapper

    nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.nmt_path, torch_dtype="auto"
    )
    llama = AutoModelForCausalLM.from_pretrained(
        config.llama_path[args.vdb_type], torch_dtype="auto"
    )
    nmt_model = KNNWrapper.break_into(nmt_model, assistant_model=llama)


def analyze_results(args):
    """Analyze results stage implementation."""
    print("Analyzing results...")
    # Implementation here


def main():
    """Main function to run the appropriate stage based on user input."""
    args = parse_main_args()
    stage_functions = {
        "create_index": create_index,
        "generate": nmt_with_lm,
        "analyze_results": analyze_results,
    }

    stage_function = stage_functions.get(args.stage)
    if stage_function:
        stage_function(args)
    else:
        print(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
