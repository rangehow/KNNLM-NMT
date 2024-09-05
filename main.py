"""A program with multi-stage processing and argument parsing."""

import argparse
import json
import os
import sys

import torch
import config
from loguru import logger


def parse_main_args():
    """Parse the main command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-stage processing program.")
    parser.add_argument(
        "stage",
        choices=["create_index", "generate", "analyze_results", "nmt"],
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

    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
    from knnlm import KNNWrapper
    from LLaMAEmbeddings import DynamicBatchSampler, DataLoader
    from torch.utils.data import Dataset
    from load import balanced_load
    from torch import float32, float16

    with torch.no_grad():
        nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.nmt_path, torch_dtype=float32
        ).to("cuda:0")

        llama = balanced_load(
            config.llama_path[args.vdb_type],
            num_devices=2,
            devices_idx=[0, 1],
            ratio=[0.8, 1],
        )

        nmt_tokenizer = AutoTokenizer.from_pretrained(config.llama_path[args.vdb_type])
        nmt_tokenizer.pad_token_id = 128004
        llm_tokenizer = AutoTokenizer.from_pretrained(config.llama_path[args.vdb_type])
        llm_tokenizer.padding_side = "left"
        llm_tokenizer.pad_token_id = 128004
        # debug
        # inputs = torch.tensor(
        #     nmt_tokenizer.encode(
        #         "Methode fÃ¼r die Berechnung der Vorhersage", add_special_tokens=False
        #     )
        #     + [128001],
        #     device=nmt_model.device,
        # ).view(1, -1)

        # logits = nmt_model(inputs)

        # result = nmt_model.generate(inputs, num_beams=5)
        # print(result, nmt_tokenizer.batch_decode(result))

        # ===============
        knn_wrapper = KNNWrapper(
            dstore_dir=f"/mnt/rangehow/knn-mt/data/{args.domain}/{config.vdb[args.vdb_type]}",
            tokenizer=llm_tokenizer,
        )
        knn_wrapper.break_into(nmt_model, assistant_model=llama)

        def my_collate(batch):
            logger.debug(f"这个batch有{len(batch)}个示例")

            return nmt_tokenizer.pad({"input_ids": batch}, return_tensors="pt")

        class TokenBatchDataset(Dataset):
            def __init__(self, texts, tokenizer):
                self.texts = [
                    tokenizer.encode(text, add_special_tokens=False) + [128001]
                    for text in texts
                ]

                self.lengths = [len(text) for text in self.texts]
                self.sorted_indices = sorted(
                    range(len(self.texts)), key=lambda i: self.lengths[i], reverse=True
                )

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                # print(idx)
                return self.texts[idx]

        with open(f"/mnt/rangehow/knn-mt/data/{args.domain}/test.de") as f:
            dataset = TokenBatchDataset(f.readlines(), nmt_tokenizer)
            sampler = DynamicBatchSampler(dataset, max_tokens=1024)
            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                shuffle=False,
                num_workers=0,
                collate_fn=my_collate,
                pin_memory=True,
            )
            translated_text = []
            from tqdm import tqdm

            for batch in tqdm(dataloader):
                batch = batch.to(nmt_model.device)
                result = nmt_model.generate(**batch, num_beams=5, max_new_tokens=250)
                decoded = nmt_tokenizer.batch_decode(result, skip_special_tokens=True)
                translated_text.extend(decoded)
                
  
            with open(
                f"data/{args.domain}/{args.vdb_type}.en", "w", encoding="utf-8"
            ) as o:
                for item in translated_text:
                    o.write(item.replace("\n", "\\n") + "\n")


def nmt_generate(args):
    """Process data stage implementation."""
    print(f"Using domain: {args.domain}")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from knnlm import KNNWrapper
    from LLaMAEmbeddings import DynamicBatchSampler, DataLoader
    from torch.utils.data import Dataset
    from load import balanced_load
    from torch import float32, float16

    nmt_model = balanced_load(
        config.nmt_path,
        encoder_decoder=True,
        devices_idx=[0, 1],
        num_devices=2,
    )

    nmt_model = torch.compile(nmt_model)
    nmt_tokenizer = AutoTokenizer.from_pretrained(config.llama_path[args.vdb_type])
    nmt_tokenizer.pad_token_id = 128004

    # debug
    # inputs = torch.tensor(
    #     nmt_tokenizer.encode(
    #         "Methode fÃ¼r die Berechnung der Vorhersage", add_special_tokens=False
    #     )
    #     + [128001],
    #     device=nmt_model.device,
    # ).view(1, -1)

    # logits = nmt_model(inputs)

    # result = nmt_model.generate(inputs, num_beams=5)
    # print(result, nmt_tokenizer.batch_decode(result))

    # ===============

    def my_collate(batch):
        logger.debug(f"这个batch有{len(batch)}个示例")

        return nmt_tokenizer.pad({"input_ids": batch}, return_tensors="pt")

    class TokenBatchDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.texts = [
                tokenizer.encode(text, add_special_tokens=False) + [128001]
                for text in texts
            ]

            self.lengths = [len(text) for text in self.texts]
            self.sorted_indices = sorted(
                range(len(self.texts)), key=lambda i: self.lengths[i], reverse=True
            )

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            # print(idx)
            return self.texts[idx]

    with open(f"/mnt/rangehow/knn-mt/data/{args.domain}/test.de") as f:
        dataset = TokenBatchDataset(f.readlines(), nmt_tokenizer)
        sampler = DynamicBatchSampler(dataset, max_tokens=1024)
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            shuffle=False,
            num_workers=0,
            collate_fn=my_collate,
            pin_memory=True,
        )
        translated_text = []
        from tqdm import tqdm

        for batch in tqdm(dataloader):
            batch = batch.to(nmt_model.device)
            result = nmt_model.generate(**batch, num_beams=5, max_new_tokens=250)
            decoded = nmt_tokenizer.batch_decode(result, skip_special_tokens=True)
            translated_text.extend(decoded)

        with open(f"data/{args.domain}/pure_nmt.en", "w", encoding="utf-8") as o:
            for item in translated_text:
                o.write(item.replace("\n", "\\n") + "\n")


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
        "nmt": nmt_generate,
        "analyze_results": analyze_results,
    }

    stage_function = stage_functions.get(args.stage)
    if stage_function:
        stage_function(args)
    else:
        print(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
