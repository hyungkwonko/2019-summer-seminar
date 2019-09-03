
# Playing Atari Pong with Deep Q Network(DQN)

## Outcome

![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/vbefore.gif)
![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/vafter.gif)

- Before training (left) and after training (right).
- You can also check [my Youtube video](https://www.youtube.com/watch?v=0UZ5U5YhGMs)

## Image preprocessing

![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/before.png)
![](https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/after.png)

Since it requires a huge amount of computational power, it is important to reduce the input size as much as possible. For this process, I made the image as 80*80 with 1 channel.

## Model
I used 3 layers of convolutional neural network architecture with the following 2 fully-connected layers. Final output is 6 numbers, where each of them has the q-value for allocated action.

## Outcome

### Loss
<img src="https://github.com/hyungkwonko/2019-summer-seminar/blob/master/project/img/loss.png" width="300">


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
- [Playing Atari with deep reinforcement learning by Mnih et al.](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Human-level control through deep reinforcement learning by Mnih et al.](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- [DQN으로 Breakout(벽돌깨기) 학습 (Korean)](https://passi0n.tistory.com/88?category=748105)
- [모두를 위한 RL (Korean)](https://www.youtube.com/watch?v=dZ4vw6v3LcA&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG)