# auto-capture
Auto capture app which opens web camera and takes a snapshot when user is ready: smiles and his/hers eyes are open.


## Installation

There are several 3th party packages are used to create this app. First you need to download `dlib` library manually ([source](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)). <br> Follow the instructors here.<br>
*  [Install](https://cgold.readthedocs.io/en/latest/first-step/installation.html) CMake.
    * **Tip**: If you are using Ubuntu, you can download CMake from Ubuntu Software.
*   Clone `dlib` to your local machine.   
    ```
    git clone https://github.com/davisking/dlib.git
    cd dlib
    python setup.py install
    ``` 

*   Clone this repo to your working directory.
*   (optional) Create and activate a virtual environment either with `pip` or conda.
*   Install necessary packages <br>
    If you are using pip.
    * `pip install -r requirements.txt`
    
    If you are using conda.
    * 
    ```
     conda create -y --name auto-capture python==3.7
     conda install --force-reinstall -y -q --name auto-capture -c conda-forge --file requirements.txt
     conda activate auto-capture
    ```



## Usage
#### WebCam Capturing
To take automated or manual shots run `capture.py` in your local device. <br>
`python capture.py`<br>
Just smile scrip will do the rest.<br>
Auto Captured photos are stored in `AutoCaps` folder. <br>
You can also take manual shots by either hitting `enter` or `space` buttons. Manual shots are stored in `ManCaps` folder.<br>
For quitting the program enter either `q` or `esc`.

#### Emotion Detection

