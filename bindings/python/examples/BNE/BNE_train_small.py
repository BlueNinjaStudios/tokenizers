from tokenizers import Tokenizer
from tokenizers.models import BNE
from tokenizers.trainers import BneTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace


# Build tokenizer
model = BNE(unk_token="[UNK]")
tokenizer = Tokenizer(model)
trainer = BneTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = ByteLevel()
files = ["data/train_small.txt"]

tokenizer.train(files, trainer)
tokenizer.save("data/bne-byte-level.json")

model.save("data/", "bne_byte_wt103")

# To implement: https://huggingface.co/docs/tokenizers/v0.20.3/en/api/models#tokenizers.models.Model
