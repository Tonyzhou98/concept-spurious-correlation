# Spurious correlation at the concept level

Paper: Explore Spurious Correlations at the Concept Level in Language Models
for Text Classification (ACL 2024)

Link: https://arxiv.org/pdf/2311.08648

### Usage

For distilbert model: ```python train_amazon_shoe.py --dataset amazon-shoe-reviews --concept size --method biased```

For llama2 model: ```python llama2_classification.py --lora_r 8 --epochs 3 --dropout 0.1 --pretrained_ckpt [llama2_checkpoint_path] --dataset amazon-shoe-reviews --concept size --method biased```

dataset name: amazon-shoe-reviews, imdb, yelp_polarity, cebab, boolq

method name: original, biased, downsample, upsample, mask
