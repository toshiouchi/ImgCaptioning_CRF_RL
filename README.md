# Improvement of n-gram repetition and caption length in GRPO reinforcement learning for image captioning models incorporating a CRF layer

## Introduction

Currently, the dominant approach for text generation involves models that utilize a Transformer Decoder in an autoregressive manner. However, autoregressive computation requires a number of operations proportional to the sequence length of the generated text, thereby demanding significant computational resources and time, as well as expensive systems. As generative AI becomes more widespread, the demand for cost-effective system implementations is expected to rise. To meet this need, it is essential to develop non-autoregressive computational methods. Non-autoregressive methods require only a single pass of computation, enabling efficient processing with lower resource consumption and reduced execution time.

In this context, we conducted supervised pre-training (SPT) on an image captioning model incorporating a CRF layer—specifically, a 
```
CLIP + MLP_Connector + BERT + CRF-layer
```
 architecture—and subsequently performed GRPO reinforcement learning based on that SPT. SPT alone resulted in n-gram repetitions and short caption lengths; however, reinforcement learning successfully reduced these repetitions and improved caption length. Notably, this model does not require iterative computation to generate captions, allowing for sufficiently fast inference.

Since the CRF layer handles token transition probabilities from sequence i-1 to sequence i, a direct calculation would entail significant computational resources and time. To address this, the vocabulary size handled by the CRF layer was restricted to the top 256 tokens based on the emission probabilities from the BERT output; this constitutes a top-k approximation for the CRF layer. Furthermore, even with the top-k approximation, the transition probability matrix remains 256 * 256 and consumes substantial resources; therefore, a low-rank setting of 32 was introduced to approximate the transition matrix as the product of two matrices (256 * 32) and (32 * 256)—representing a low-rank approximation.

I originally developed the program about a year ago, so it was designed for the Hugging Face `transformers` v4 series; however, now that I have some free time following a break in the computations, I have updated it to support the `transformers` v5 series and am running the calculations.

## SPT results

Regarding SPT,

https://qiita.com/toshiouchi/items/cef093035db06cc54e9a

might be a useful reference. I have recalculated it to ensure compatibility with the transformers 5.x series.

### Values ​​of each score for the test data

```
cider           0.919
rougeL          0.514
clip_score      0.280
bert_score      0.837
repeat_count    2.26
length_penalty -1.70 * e-3
```

The `repeat_count` takes a high value when there are many n-gram repetitions and a low value when there are few.

I will explain `length_penalty`. The lengths of the generated caption and the ground-truth caption are scaled to 1 by dividing them by a fixed maximum caption length of 97. `length_penalty` is the average value of the squared difference between these two scaled lengths, multiplied by -1. A value closer to 0 indicates that the length of the generated caption is closer to that of the ground-truth caption.

### Genarated captions

Picture 1
![Screenshot from 2026-07-01 09-55-21.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/3bbceb1c-0f57-47b8-aa16-e03e919fbeec.png)
```
hypo: [CLS] a dog that is playing its floor with its floor. [SEP]
refe: [CLS] a golden retriever eating something on a hardwood floor. [SEP]
```

Picture 2
![Screenshot from 2026-07-01 09-57-15.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/89b83647-d5d1-4604-8da7-a324a7ad4f0a.png)
```
hypo: [CLS] a white plate topped with a plate on a plate. [SEP]
refe: [CLS] a plate with a couple of sandwiches on it [SEP]
```

```
hypo: [CLS] a dog that is playing its floor with its floor. [SEP]
refe: [CLS] a golden retriever eating something on a hardwood floor. [SEP]
hypo: [CLS] a pair of scissors and scissors and a table. [SEP]
refe: [CLS] a selection of scissors, including pinking shears, under a glass shelf. [SEP]
hypo: [CLS] a woman holding a pink using under an umbrella. [SEP]
refe: [CLS] a woman holds up an electronic cigarette underneath an umbrella. [SEP]
hypo: [CLS] a large clock tower front of a clock. [SEP]
refe: [CLS] a steeple with a bell tower and a clock with many windows. [SEP]
hypo: [CLS] two tour busses parked in a parking lot. [SEP]
refe: [CLS] two blue buses parked in front of a volvo store. [SEP]
hypo: [CLS] a soccer throwing a soccer on a ball. [SEP]
refe: [CLS] a soccer player runs up to kick the ball while the crowd watches. [SEP]
hypo: [CLS] two are benches that are benches park bench. [SEP]
refe: [CLS] a couple of creepy statues sitting on a bench in a park at night. [SEP]
hypo: [CLS] a woman in a tennis racquet. [SEP]
refe: [CLS] the female tennis player swings the racket overhead. [SEP]
hypo: [CLS] two giraffes are standing in a grassy field. [SEP]
refe: [CLS] a couple of giraffe playing around in a field of grass. [SEP]
hypo: [CLS] a bird perched being birds near a window. [SEP]
refe: [CLS] a bird on the outside of a door with clear windows [SEP]
hypo: [CLS] a woman in a baseball bat holding a ball. [SEP]
refe: [CLS] a woman dressed in uniform swinging a baseball bat. [SEP]
hypo: [CLS] a baseball game baseball players on a baseball game. [SEP]
refe: [CLS] a baseball game in progress, with many onlookers. [SEP]
hypo: [CLS] two truck are in the back of a car truck. [SEP]
refe: [CLS] a truck with a cage in the bed carries live animals on a highway. [SEP]
hypo: [CLS] a baseball player swinging a bat at a baseball game. [SEP]
refe: [CLS] a group of men in a field playing baseball. [SEP]
hypo: [CLS] a cutting board let laid differentni on top tools let squash fresh
refe: [CLS] various vegetables sit openly on the counter top. [SEP]
hypo: [CLS] a black and white cat laying on a couch. [SEP]
refe: [CLS] a long haired cat that is posing for a picture. [SEP]
hypo: [CLS] a man on a motorcycle on a motorcycle. [SEP]
refe: [CLS] a person in motorcycle apparel from the back, standing in front of a person in motorcycle apparel on a motorcycle from the side, on pave surface by water. [SEP]
hypo: [CLS] a baseball player swinging a bat at a baseball game. [SEP]
refe: [CLS] baseball player completing his batting swing at home plate [SEP]
hypo: [CLS] a man with a dog on a dog boat dog. [SEP]
refe: [CLS] a person that is floating in some water [SEP]
hypo: [CLS] a cutting board with carrot chopped cutting board multiple knife. [SEP]
refe: [CLS] carrots and cucumber on wooden cutting board near knives. [SEP]
hypo: [CLS] a donuts donuts somekin of a table. [SEP]
refe: [CLS] a cook book for making donuts with donuts and coffee pictures on it ' s cover. [SEP]
hypo: [CLS] a bathroom with a toilet in a toilet. [SEP]
refe: [CLS] a white toilet sitting underneath a window next to a tub. [SEP]
hypo: [CLS] a cat sitting on top of a park bench. [SEP]
refe: [CLS] a bird sitting on top of a bench and a cat sitting underneath it. [SEP]
hypo: [CLS] a woman in a kitchen with a kitchen. [SEP]
refe: [CLS] the frantic lady is checking her latest concoction. [SEP]
```

## Results of GRPO Reinforcement Learning

### Learning curve for validation data

The learning curves for repeat_penalty, length_penalty × 1000, CIDEr, ROUGE-L, CLIP_score, and BERT_score during validation are shown.

![Screenshot from 2026-07-01 10-01-49.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/787c86ae-b3af-4d0c-9c2b-1f7f222eef1b.png)

![Screenshot from 2026-07-01 10-05-08.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/2ce7cf88-ae41-474f-8098-b6fcae5edb32.png)

![Screenshot from 2026-07-01 10-05-31.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/d543a7b5-f66d-4dff-9403-e10ff8973e1f.png)

![Screenshot from 2026-07-01 10-06-04.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/7ab441ed-ed94-42a7-a4e2-90701cd554e1.png)

![Screenshot from 2026-07-01 10-06-24.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/99b0d57f-23bf-4b4a-b518-33b047620e26.png)

![Screenshot from 2026-07-01 10-06-43.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/8e47ddd1-9718-4736-9ac8-edd34040e9b8.png)


### Each score for the test data

```
cider           0.861
rougeL          0.526
clip_score      0.275
bert_score      0.833
repeat_count    0.240
length_penalty -1.34 e-3
```

### Generated captions

Picture 1
![Screenshot from 2026-07-01 09-55-21.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/02b0d709-4739-4481-9a1e-dd8c13b7d7f1.png)
```
hypo: [CLS] a dog that is sitting on the floor. [SEP]
refe: [CLS] a dog chewing on a stick on a hardwood floor. [SEP]
```

Picture 2
![Screenshot from 2026-07-01 09-57-15.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/89b83647-d5d1-4604-8da7-a324a7ad4f0a.png)
```
hypo: [CLS] a close up topped with a sandwich. [SEP]
refe: [CLS] a plate with a couple of sandwiches on it [SEP]
```

```
hypo: [CLS] a dog that is sitting on the floor. [SEP]
refe: [CLS] a dog chewing on a stick on a hardwood floor. [SEP]
hypo: [CLS] a wooden crafts sci hung sitting on a table. [SEP]
refe: [CLS] pairs of scissors and a rubber mallet laid out on a table [SEP]
hypo: [CLS] a person standing in an open holding umbrella. [SEP]
refe: [CLS] a woman staying dry from the rain and holding an umbrella. [SEP]
hypo: [CLS] a large clock tower on top of a building. [SEP]
refe: [CLS] tall tower with a large clock in the center of the tower [SEP]
hypo: [CLS] a couple of parked in a parking lot. [SEP]
refe: [CLS] two blue buses parked in front of a volvo store. [SEP]
hypo: [CLS] a soccer throwing a soccer on the ball. [SEP]
refe: [CLS] a soccer player runs up to kick the ball while the crowd watches. [SEP]
hypo: [CLS] a couple of benches that are benches clock. [SEP]
refe: [CLS] a couple of creepy statues sitting on a bench in a park at night. [SEP]
hypo: [CLS] a woman in a tennis racquet. [SEP]
refe: [CLS] woman tennis player twisting both arms to snag the ball [SEP]
hypo: [CLS] two giraffes are standing in a grassy field. [SEP]
refe: [CLS] two very pretty giraffes together in a big grassy field. [SEP]
hypo: [CLS] a large sitting on top of a window. [SEP]
refe: [CLS] a bird on the outside of a door with clear windows [SEP]
hypo: [CLS] a woman that is ready ready at the ball. [SEP]
refe: [CLS] a female baseball player looks on with intensity while swinging the bat. [SEP]
hypo: [CLS] a group of people on wait home played. [SEP]
refe: [CLS] the pitcher just threw the pitch at the baseball game. [SEP]
hypo: [CLS] a couple of in the back of a car truck. [SEP]
refe: [CLS] a couple of animals are in the back of a truck [SEP]
hypo: [CLS] a baseball player swinging a bat while home. [SEP]
refe: [CLS] a baseball game is in action at the plate. [SEP]
hypo: [CLS] a view onionsdis least differentni on a table. [SEP]
refe: [CLS] there are some fresh vegetables on a cutting board [SEP]
hypo: [CLS] a black and white on a leatherless. [SEP]
refe: [CLS] a cat that has very long and white whiskers. [SEP]
hypo: [CLS] a man on a motorcycle on a motorcycle. [SEP]
refe: [CLS] a man sitting on a motorcycle as another man walks towards him. [SEP]
hypo: [CLS] a baseball player is ready at the ball. [SEP]
refe: [CLS] baseball player completing his batting swing at home plate [SEP]
hypo: [CLS] a man is sitting on top of water. [SEP]
refe: [CLS] a person that is floating in some water [SEP]
hypo: [CLS] a cutting board with carrot chopped cutting board. [SEP]
refe: [CLS] a close up of many different vegetables on a cutting board. [SEP]
hypo: [CLS] a close up doughnut and a table. [SEP]
refe: [CLS] a spanish advertisement for two sweet glazed donuts. [SEP]
hypo: [CLS] a bathroom with a toilet in the floor. [SEP]
refe: [CLS] a toilet that has cans of paint next to it. [SEP]
hypo: [CLS] a cat is sitting on a park bench. [SEP]
refe: [CLS] a bird perched on a park bench as a cat hides underneath. [SEP]
hypo: [CLS] a woman standing in a kitchen preparing oven. [SEP]
refe: [CLS] a person in a room in front of a stove. [SEP]
```
## Comparison

Reinforcement learning has improved `repeat_count` and `length_penalty`. Language-specific scores and `clip_score` are also maintained at favorable levels.

## I'm putting the reinforcement learning program on GitHub.

https://github.com/toshiouchi/ImgCaptioning_CRF_RL/tree/main

The directories are GRPO_suppl_compatible_transformers5 and COCO_SPT_compatible_transformers5.

The training data is COCO train2017.

## Points to note regarding the program.

1. We use a proprietary stochastic Viterbi sampling function developed for GRPO.
1. The image captioning model performs calculations involving top-k approximation in three stages: sampling, training, and kl_div reference. The `torch.topk` function is executed only during sampling; for the training and reference stages, the `beam_emission_scores` are calculated using the `top_indices` obtained during sampling.
1. The token paths used to derive `sampled_log_probs` from `log_probs` are also the same token paths used during sampling.
1. When calculating the reference log-probabilities, information regarding the tokens from sequence *i*-1 is required to compute the log-probabilities for sequence *i*; the tokens for sequence *i*-1 are also those obtained through sampling. (Function: `_compute_ref_probs`)

