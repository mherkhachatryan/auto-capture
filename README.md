# auto-capture
Auto capture app which opens web camera and takes a snapshot when user is ready: smiles and his/hers eyes are open.


## Installation

Several 3th party packages are used to create this app. First you need to download `dlib` library manually ([source](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)). <br> To do this follow the instructors here.<br>
*  [Install](https://cgold.readthedocs.io/en/latest/first-step/installation.html) CMake.
    * **Tip**: If you are using Ubuntu, you can download CMake from Ubuntu Software.
*   Clone `dlib` to your local machine.   
    ```
    git clone https://github.com/davisking/dlib.git
    cd dlib
    python setup.py install
    ``` 

*   Clone this repo to your working directory.
    ```
    git clone https://github.com/mherkhachatryan/auto-capture.git
    cd auto-capture/
    ```
*   Make sure you are using `python 3.7`
*   (optional) Create and activate a virtual environment either with `pip` or `conda`.
*   Install necessary packages. <br>
    If you are using pip.
    * `pip install -r requirements.txt`
    
    If you are using conda.
    * 
    ```
     conda create -y --name auto-capture python==3.7
     pip install -r requirements.txt
     conda activate auto-capture
    ```



## Usage
#### WebCam Capturing
To take automated or manual shots run `capture.py` in your local device. <br>
```
python capture.py
```
Just smile: scrip will do the rest.<br>
Auto Captured photos are stored in `AutoCaps` folder. <br>
You can also take manual shots by either hitting `enter` or `space` buttons. Manual shots are stored in `ManCaps` folder.<br>
Available options to change photo location, show contour on facial landmarks, or change default value of 
EAR and MAR threshold. Just refer to scrip help by <br>
```
python capture.py --help
```
For quitting the program enter either `q` or `esc`.

###### Example 
```
python capture.py -a '/home/mher/Downloads' -v True
```
This example will show contour around eyes and mouth, and will save automated taken picture in `/home/mher/Downloads`
#### Emotion Detection
Use this script to detect your emotion on taken photos. Image path is required.<br>
###### Example 
```
python emotion_detector.py "home/mher/image.png" 
```
This example takes a `image.png` and will print emotion detected on face, containing in the image.

## Code Overview
*code overview and method explanation are coming later*