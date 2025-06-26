from tokenizers import Tokenizer
from tokenizers.models import models, Model
from tokenizers import trainers
from tokenizers.pre_tokenizers import ByteLevel

for e in dir(trainers):
    print(e)

tokenizer = Tokenizer(models.BNE(unk_token="[UNK]"))
trainer = trainers.BneTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = ByteLevel()
files = ["data\wiki.train.tokens"]

print(tokenizer.pre_tokenizer.pre_tokenize)

tokenizer.train(files, trainer)
tokenizer.save("data/bne-byte-level.json")

# To implement: https://huggingface.co/docs/tokenizers/v0.20.3/en/api/models#tokenizers.models.Model
