FROM python:3.10
WORKDIR //Users/nilaygaitonde/Documents/Projects/JANkari_IronGolem
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD "echo hi we're done copying"
EXPOSE 8051
CMD "looks like the port has been exposed"
CMD ["streamlit","run","Streamlit/app.py","--server.port=8051","--server.address=0.0.0.0"]
