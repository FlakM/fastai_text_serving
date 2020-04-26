# About

This is a simple showdown of serving model created in this colab: https://colab.research.google.com/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb

I wasn't able to find any examples of serving text models (only partial examples of computer vision ones)

It uses plain pkl models exported using colab code.

Usefull links:

* https://course.fast.ai/videos/?lesson=4
* https://forums.fast.ai/t/deep-learning-lesson-3-notes/29829



# Set me up

```bash
# https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
virtualenv -p python3 $HOME/tmp/fastai/
pip3 install -r requirements.txt
```


# Running

```
source $HOME/tmp/fastai/bin/activate
python3 src/server.py
```

To test service run:

```
curl -X POST -F 'text=the movie was great. Acting was superb' 127.0.0.1:4000/predict
```

