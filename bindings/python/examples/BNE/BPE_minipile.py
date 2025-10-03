from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
import datasets

# Build tokenizer
model = BPE(unk_token="[UNK]")
tokenizer = Tokenizer(model)
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = ByteLevel()

# Load dataset
dataset = datasets.load_dataset("JeanKaddour/minipile", split="train")

# Build an iterator over this dataset
def batch_iterator():
    batch_size = 1000
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]


print(tokenizer.pre_tokenizer.pre_tokenize)

tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(dataset))
tokenizer.save("data/bpe_byte-level_minipile.json")

model.save("data/", "bpe_byte-level_minipile_M")

