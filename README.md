# Code for Misleading Metadata Detection on YouTube

Note: The directories inside the final_data directory need to contain the crawled data files first. 

To train/test UCNet, run

```
python3 LSTM_Comments_Avg_Net.py
```

To train/test simple classifiers, run

```
python3 simple_classifiers.py
```


To run our implementation of the baseline, first uncomment lines 28-30 in TweetClassifier/constantsTweet.py and then run

```
python3 baseline.py
```