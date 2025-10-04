
from tokenizers import Tokenizer
from tokenizers.models import BNE
from tokenizers.trainers import BneTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
import datasets
import os

# Define name of test
name = "bne_byteLevel_minipile_5_2"

# Build tokenizer
model = BNE(unk_token="[UNK]")
tokenizer = Tokenizer(model)
trainer = BneTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = ByteLevel()

# Load dataset
dataset = datasets.load_dataset("JeanKaddour/minipile", split="train").shard(num_shards=20, index=7) #.train_test_split(test_size=0.75, seed=42)["train"]

# Build an iterator over this dataset
def batch_iterator():
    batch_size = 1000
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]

os.mkdir("data/" + name)

tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(dataset))
tokenizer.save(f"data/{name}/{name}.json")

model.save(f"data/{name}/", name)

