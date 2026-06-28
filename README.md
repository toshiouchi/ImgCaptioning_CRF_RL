# Improvement of n-gram repeats and CIDEr language scores in GRPO reinforcement learning for image captioning models incorporating a CRF layer

We performed supervised learning （SFT) on an image captioning model equipped with a CRF layer,
```
CLIP + MLP_connector + BERT + CRF-layer
```
and subsequently applied GRPO reinforcement learning to the results. The results of SFT were characterized by n-gram repetitions and relatively short sentences. We report here that, by employing reinforcement learning, we were able to reduce n-gram repetitions and improve the language score.

## Result of SFT

### Each score based on test data

```
cider        0.505
rougeL       0.430
clip_score   0.255
bert_score   0.808
repeat_count 1.21
```

### Generated captions

Picture 1
![Screenshot from 2026-06-14 08-22-09.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/adf35ffa-19bd-4952-a8c0-908883156702.png)
```
hypo: [CLS] a pair of scissors three glue from tape board. [SEP]
refe: [CLS] a selection of scissors, including pinking shears, under a glass shelf. [SEP]
```

Picture 2
![Screenshot from 2026-06-14 08-22-30.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/37fdf49f-75bd-42e5-8c38-d9216604637e.png)
```
hypo: [CLS] a person holding an umbrella on a kite. [SEP]
refe: [CLS] a person holding an open umbrella and a cell phone [SEP]
```

```
hypo: [CLS] a woman is holding a baseball yankees uniform. [SEP]
refe: [CLS] a woman preparing to swing a baseball bat. [SEP]
hypo: [CLS] a group of a carriagely decorated line pictures. [SEP]
refe: [CLS] four horses are pulling a small green carriage. [SEP]
hypo: [CLS] a book standing in front of character carnival hand. [SEP]
refe: [CLS] a person carrying some bags looking at a bunch of green benches. [SEP]
hypo: [CLS] a person laptop held front of a table. [SEP]
refe: [CLS] beautiful woman behind a laptop computer at a white table. [SEP]
hypo: [CLS] a fire hydrant in front of a fire hydrant it. [SEP]
refe: [CLS] a large fire hydrant on a truck [SEP]
hypo: [CLS] a fire hydrant sitting on a painting. [SEP]
refe: [CLS] a water hydrant on a pavement and a potted plant [SEP]
hypo: [CLS] a sheep standing next to a field eating. [SEP]
refe: [CLS] sheep eat from a blue bowl with the sheepdog behind them. [SEP]
hypo: [CLS] a large bus parked shape curve engine area. [SEP]
refe: [CLS] a green and light blue bus traveling down a highway. [SEP]
hypo: [CLS] a sandwich of bread item sitting on a plate. [SEP]
refe: [CLS] a white plate that has a sandwich cut in half [SEP]
hypo: [CLS] a cake shaped cookies sitting on a cake spell. [SEP]
refe: [CLS] cake on a plate with a bottle of milk that has a straw [SEP]
hypo: [CLS] a group of people on the cockpit carousel board aircraft
refe: [CLS] a group of people getting on a small plane. [SEP]
hypo: [CLS] a woman is in front of a desk. [SEP]
refe: [CLS] a women leans over her desk towards a laptop. [SEP]
hypo: [CLS] a woman and a woman chair patient game. [SEP]
refe: [CLS] two people seated on a couch, one with glasses and holding remotes. [SEP]
hypo: [CLS] a close up of a cat an e opencake ribbon. [SEP]
refe: [CLS] a cat wearing a holiday hat reclines on an blanket while being petted. [SEP]
hypo: [CLS] a living room with a red upstairs a room. [SEP]
refe: [CLS] there is a beige couch and orange pillows in this living room [SEP]
hypo: [CLS] a baby ya brows standing in a field. [SEP]
refe: [CLS] several cows laying in the grass on a sunny day. [SEP]
hypo: [CLS] a person on a surfboard being wavy pool board. [SEP]
refe: [CLS] a man on a surfboard on the water. [SEP]
hypo: [CLS] a war signs tv telling and " says  five falling [SEP]
refe: [CLS] a pole filled with many various city street signs. [SEP]
hypo: [CLS] a white bus parked amongst on the street. [SEP]
refe: [CLS] a bus is driving down a busy road. [SEP]
hypo: [CLS] a person of food turkey take in front of food. [SEP]
refe: [CLS] two people standing in a kitchen preparing food. [SEP]
hypo: [CLS] a large on a largeran water cars. [SEP]
refe: [CLS] several boats dry docked on a pier in a harbor area. [SEP]
hypo: [CLS] a bowl ordered biscuitsstick plates some food. [SEP]
refe: [CLS] meat and vegetables are in a white bowl. [SEP]
hypo: [CLS] a large sided roof out numbers kind golden eagle. [SEP]
refe: [CLS] well lit skyscrapers and a clock tower in the evening. [SEP]
hypo: [CLS] a pizza sitting on top cheese next tocho par. [SEP]
refe: [CLS] a small pizza with several ingredients missing one slice. [SEP]
hypo: [CLS] a close up of a wash airplane mirror. [SEP]
refe: [CLS] the toilet is empty and the seat is up [SEP]
hypo: [CLS] a surfboardd in a washed waters washed background. [SEP]
refe: [CLS] an aerial view of two people on surfboards in the water. [SEP]
hypo: [CLS] a large clock tower on the university next to billboard pole. [SEP]
refe: [CLS] an advertisement is seen on the side of a very tall building. [SEP]
hypo: [CLS] a cow group animals showing sheep mountain eating. [SEP]
refe: [CLS] several cows grazing in a field under a tree. [SEP]
hypo: [CLS] a cow made corn bull cow farmduri. [SEP]
refe: [CLS] a long horned cow standing next to two decorative circles [SEP]
hypo: [CLS] a meter towers tv railway posted between model writing wall. [SEP]
refe: [CLS] a north bound trains sign pointing to the left in a station. [SEP]
hypo: [CLS] a woman in her looked eat industrial eat industrial eat. [SEP]
refe: [CLS] people in line being served food by a woman behind the counter. [SEP]
hypo: [CLS] a smallp swan on a large next to marinafront. [SEP]
refe: [CLS] a man, woman and two dogs are moving along in a motor boat. [SEP]
hypo: [CLS] a woman standing in front of a room. [SEP]
refe: [CLS] a room with a woman standing in front of a television playing a video game. [SEP]
hypo: [CLS] an doll blend has print six pipe clock. [SEP]
refe: [CLS] a white toilet with a wooden seat on black tile floor. [SEP]
hypo: [CLS] a person player jumped ping hit a ball. [SEP]
refe: [CLS] a man running across a tennis court holding a racquet. [SEP]
hypo: [CLS] a train that is on a train railway. [SEP]
refe: [CLS] the platform and tracks at a railroad intersection [SEP]
hypo: [CLS] a rugbyor rugby mid raises something ball. [SEP]
refe: [CLS] this is two men in a soccer match [SEP]
hypo: [CLS] a baseball player that is base signing turn. [SEP]
refe: [CLS] a baseball player winding up to swing the bat. [SEP]
hypo: [CLS] a woman is in front of a motorcycle circus. [SEP]
refe: [CLS] a women stands next to a yellow motorcycle. [SEP]
hypo: [CLS] a bird bird sitting on a tree bird. [SEP]
refe: [CLS] a bird flying towards a branch on tree. [SEP]
```

## Reinforcement Learning via GRPO

### Learning curve for validation data

The learning curves for validation repeat penalty, CIDEr, ROUGE-L, CLIP score, and BERTScore are shown.

![Screenshot from 2026-06-27 14-15-43.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/05201ec4-3fd0-4485-8978-5ff088922f72.png)

![Screenshot from 2026-06-27 14-16-14.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/9c1cae85-5171-4f3f-be29-deb3c96bc131.png)

![Screenshot from 2026-06-27 14-16-36.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/fe6d0b76-619d-43aa-b1ed-d0c902246e08.png)

![Screenshot from 2026-06-27 14-17-02.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/7d776900-a3ab-4d78-a2a2-54894b28e973.png)

![Screenshot from 2026-06-27 14-18-25.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/032504f3-d897-49bd-a602-d3c6be06eda6.png)

![Screenshot from 2026-06-27 14-18-46.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2958180/8fb47071-1a6b-48a4-859f-82f3a57dd367.png)


### Each score of the test data

```
cider        0.741
rougeL       0.494
clip_score   0.267
bert_score   0.826
repeat_count 0.272
```

### Generated captions

Picture 1
```
hypo: [CLS] a pair of scissors sitting on a table. [SEP]
refe: [CLS] a selection of scissors, including pinking shears, under a glass shelf. [SEP]
```

Picture 2
```
hypo: [CLS] a person holding an umbrella on the street. [SEP]
refe: [CLS] a person holding an open umbrella and a cell phone [SEP]
```

```
hypo: [CLS] a woman is getting ready to hit a ball. [SEP]
refe: [CLS] a woman preparing to swing a baseball bat. [SEP]
hypo: [CLS] a group of a horses in a horses. [SEP]
refe: [CLS] four horses are pulling a small green carriage. [SEP]
hypo: [CLS] a man sitting on a sidewalk ball it. [SEP]
refe: [CLS] a person carrying some bags looking at a bunch of green benches. [SEP]
hypo: [CLS] a woman is in front of a table. [SEP]
refe: [CLS] beautiful woman behind a laptop computer at a white table. [SEP]
hypo: [CLS] a fire truck parked fire hydrant city street. [SEP]
refe: [CLS] a large fire hydrant on a truck [SEP]
hypo: [CLS] a red fire hydrant sitting on a sidewalk. [SEP]
refe: [CLS] a water hydrant on a pavement and a potted plant [SEP]
hypo: [CLS] a couple of sheep graz milk bucket together. [SEP]
refe: [CLS] sheep eat from a blue bowl with the sheepdog behind them. [SEP]
hypo: [CLS] a double decker bus driving down the street. [SEP]
refe: [CLS] a green and light blue bus traveling down a highway. [SEP]
hypo: [CLS] a close up sandwich next to a plate. [SEP]
refe: [CLS] a white plate that has a sandwich cut in half [SEP]
hypo: [CLS] a frosted arranged berries straw variety eaten. [SEP]
refe: [CLS] cake on a plate with a bottle of milk that has a straw [SEP]
hypo: [CLS] a group of people takeoff front of an airport. [SEP]
refe: [CLS] a group of people getting on a small plane. [SEP]
hypo: [CLS] a woman sitting in front of a desk. [SEP]
refe: [CLS] a women leans over her desk towards a laptop. [SEP]
hypo: [CLS] a man and a woman couch to video game. [SEP]
refe: [CLS] two people seated on a couch, one with glasses and holding remotes. [SEP]
hypo: [CLS] a close upno tired stares eyed purse tus. [SEP]
refe: [CLS] a cat wearing a holiday hat reclines on an blanket while being petted. [SEP]
hypo: [CLS] a living room with a couch and a table. [SEP]
refe: [CLS] there is a beige couch and orange pillows in this living room [SEP]
hypo: [CLS] a cow cows by standing in the grass. [SEP]
refe: [CLS] several cows laying in the grass on a sunny day. [SEP]
hypo: [CLS] a man riding a surf board on the water. [SEP]
refe: [CLS] a man on a surfboard on the water. [SEP]
hypo: [CLS] a couple of sign next to a street. [SEP]
refe: [CLS] a pole filled with many various city street signs. [SEP]
hypo: [CLS] a white bus driving down a city street. [SEP]
refe: [CLS] a bus is driving down a busy road. [SEP]
hypo: [CLS] a person cooking just apron let while pizza ball. [SEP]
refe: [CLS] two people standing in a kitchen preparing food. [SEP]
hypo: [CLS] a group of docked parked in the water. [SEP]
refe: [CLS] several boats dry docked on a pier in a harbor area. [SEP]
hypo: [CLS] a close up filled with meat and vegetables. [SEP]
refe: [CLS] meat and vegetables are in a white bowl. [SEP]
hypo: [CLS] a clock tower on the side of a building. [SEP]
refe: [CLS] well lit skyscrapers and a clock tower in the evening. [SEP]
hypo: [CLS] a pizza covered topped sitting on a plate. [SEP]
refe: [CLS] a small pizza with several ingredients missing one slice. [SEP]
hypo: [CLS] a close up in front of a bathroom. [SEP]
refe: [CLS] the toilet is empty and the seat is up [SEP]
hypo: [CLS] a couple of people ocean on the water. [SEP]
refe: [CLS] an aerial view of two people on surfboards in the water. [SEP]
hypo: [CLS] vehicles monument worker intersection the air city street. [SEP]
refe: [CLS] an advertisement is seen on the side of a very tall building. [SEP]
hypo: [CLS] a group of sheep standing in the grass. [SEP]
refe: [CLS] several cows grazing in a field under a tree. [SEP]
hypo: [CLS] a cow cows in front of a cow. [SEP]
refe: [CLS] a long horned cow standing next to two decorative circles [SEP]
hypo: [CLS] a close up operated sitting on the street. [SEP]
refe: [CLS] a north bound trains sign pointing to the left in a station. [SEP]
hypo: [CLS] a group of people are food picnic pit man. [SEP]
refe: [CLS] people in line being served food by a woman behind the counter. [SEP]
hypo: [CLS] a couple of people sitting on a boat. [SEP]
refe: [CLS] a man, woman and two dogs are moving along in a motor boat. [SEP]
hypo: [CLS] two women playing woman in a video game. [SEP]
refe: [CLS] a room with a woman standing in front of a television playing a video game. [SEP]
hypo: [CLS] a white toilet sits on a wooden toilet. [SEP]
refe: [CLS] a white toilet with a wooden seat on black tile floor. [SEP]
hypo: [CLS] a young man about to hit a tennis ball. [SEP]
refe: [CLS] a man running across a tennis court holding a racquet. [SEP]
hypo: [CLS] the empty moving night tracks way way way through. [SEP]
refe: [CLS] the platform and tracks at a railroad intersection [SEP]
hypo: [CLS] a group of people playing on the ball. [SEP]
refe: [CLS] this is two men in a soccer match [SEP]
hypo: [CLS] a baseball player another about to a field. [SEP]
refe: [CLS] a baseball player winding up to swing the bat. [SEP]
hypo: [CLS] a woman standing in front of a motorcycle. [SEP]
refe: [CLS] a women stands next to a yellow motorcycle. [SEP]
hypo: [CLS] a bird bird sitting on a tree branch. [SEP]
refe: [CLS] a bird flying towards a branch on tree. [SEP]
```

## I have uploaded the program used for reinforcement learning.

Directory is GRPO_suppl_compatible_transformers5
