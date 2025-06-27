import datasets

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

# Build a tokenizer

bne_tokenizer = Tokenizer(models.BNE())
bne_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
bne_tokenizer.normalizer = normalizers.Lowercase()

# Initialize a dataset
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")


# Build an iterator over this dataset
def batch_iterator():
    batch_size = 1000
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]


# And finally train
bne_tokenizer.train_from_iterator(batch_iterator(), length=len(dataset))
bne_tokenizer.save("data/bne_byte_wikitext-103.json")

# tokenizer = Tokenizer(models.BNE(
#    join(args.out, "{}-vocab.json".format(args.name)),
#    join(args.out, "{}-merges.txt".format(args.name)),
#    add_prefix_space=True,
# ))


print(bne_tokenizer.encode("Training ByteLevel BPE is very easy").tokens)
