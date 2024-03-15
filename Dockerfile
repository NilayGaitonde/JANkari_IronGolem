FROM python:3.10
WORKDIR //Users/nilaygaitonde/Documents/Projects/JANkari_IronGolem
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 3000
CMD ["streamlit","run","Streamlit/app.py","--server.port=8051","--server.address=0.0.0.0"]
