# Create a virtual environment 
virtualenv --no-site-packages -p python3 ~/DeepFaceDrawing

# Activate the virtual environment
source ~/DeepFaceDrawing/bin/activate

# download DeepFaceDrawing code
git clone https://github.com/IGLICT/DeepFaceDrawing-Jittor

# Update pip
pip install -U pip

# install necessary libraries
cd DeepFaceDrawing-Jittor
pip install jittor==1.1.7.0
pip install pyqt5==5.9.2
pip install Pillow
pip install scipy
pip install dominate
pip install opencv-python==4.1.0.25
mv ./heat/bg.jpg ./heat/.jpg

# download pretrained model 
cd ./Params
wget https://www.dropbox.com/s/5s5c4zuq6jy0cgc/Combine.zip
unzip Combine.zip && rm Combine.zip
wget https://www.dropbox.com/s/cs4895ci51h8xn3/AE_whole.zip
unzip AE_whole.zip && rm AE_whole.zip

# run code
cd ..
python3.7 test_model.py