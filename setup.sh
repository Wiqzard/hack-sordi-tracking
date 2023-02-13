
python -m pip install paddlepaddle-gpu==2.4.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
#git clone ...
cd PaddleDetection
pip install -r requirements.txt
python setup.py install
cd ../

pip install Cython
cd bbox
python3 setup.py build_ext --inplace
pyximport.install() 
pyximport.install(setup_args={"script_args" : ["--verbose"]})
cd ../
%cd {HOME}
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack && pip3 install -q -r requirements.txt
cd ByteTrack && python3 setup.py -q develop
pip install -q cython_bbox
pip install -q onemetric
cd ../
cd tracking-tools && git pull