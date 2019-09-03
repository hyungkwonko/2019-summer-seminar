
# Playing Atari Pong with Deep Q Network(DQN)

![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/vbefore.gif)
![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/vafter.gif)


## Image preprocessing

![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/before.png)
![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/after.png)

Since it requires a huge amount of computational power, it is important to reduce the input size as much as possible. For this process, I made the image as 80*80 with 1 channel.

## Model
I used 3 layers of convolutional neural network architecture with the following 2 fully-connected layers. Final output is 6 numbers, where each of them has the q-value for allocated action.

## Outcome

### Loss
![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/loss.png)

![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/loss2.png)

- The loss is not linearly decreasing and it was hard to find any latent pattern regarding it.

### Average Max Q value
![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/avgmaxq.png)

### Average reward for recent 50 episodes
![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/avgrwd.png)

### Step for each episode
![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/avgstep.png)

## Dependency
To run this code, I used `python 3.7.3` with `tensorflow 1.14.0` and `tensorflow-gpu 1.13.1`.


## References
- [Naver D2: Introduction of Deep Reinforcement Learning by Donghyun Kwak (Korean)](https://www.youtube.com/watch?v=dw0sHzE1oAc)
- [RL Korea: How to study Reinforcement Learning (Korean)](https://github.com/reinforcement-learning-kr/how_to_study_rl)
- [Stanford CS234: Reinforcement Learning by Emma Brunskill | Winter 2019](https://www.youtube.com/watch?v=FgzM3zpZ55o&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
- [MIT 6.S191: Introduction to Deep Learning by Alexander Amini | Winter 2019](https://www.youtube.com/watch?v=i6Mi2_QM3rA&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=5)
