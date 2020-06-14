FROM python:3.7

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

# Clone the Detectron repository
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN pip install -e detectron2

# install the pycocotools
RUN python -m pip install pycocotools==2.0.0

EXPOSE 8080

COPY . /app

CMD streamlit run --server.port 8080 --server.enableCORS false app.py
