FROM lambci/lambda:build-python3.7

COPY requirements.txt .

RUN which pip
RUN pip install --upgrade pip

#RUN conda install --yes --file requirements.txt 

#RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html


#RUN pip freeze


RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
