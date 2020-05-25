# auto-capture
Auto capture app which opens web camera and takes a snapshot when user is ready: smiles and his/hers eyes are open.

#### 3th party apps

First detecting face from image, using `face_detection` library. 
for itt istallation we need dlib v19.9. For dlib you need to [install](https://cgold.readthedocs.io/en/latest/first-step/installation.html) CMake.


Steps to install  `dlib`([source](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf) )<br> 
`git clone https://github.com/davisking/dlib.git` <br>
```
cd dlib
python setup.py install
``` 
Then install `face_recognition` package for more details refer [here](https://github.com/ageitgey/face_recognition). <br>
```
pip install face_recognition
```

https://askubuntu.com/questions/80448/what-would-cause-the-gi-module-to-be-missing-from-python
to install gi for playing sound 
