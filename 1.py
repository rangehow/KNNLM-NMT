from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import torch

mname = "facebook/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)
model = torch.compile(model)


def pre_forward_hook(module, input):

    print("here is hook")

model.register_forward_pre_hook(pre_forward_hook)
model.model.register_forward_pre_hook(pre_forward_hook)
input = "Machine learning is great, isn't it?"
input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)  # Maschinelles Lernen ist großartig, oder?
