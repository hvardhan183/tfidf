FROM ubuntu:18.04
RUN apt-get update && apt-get install -y \
   python3-pip python3-dev
COPY /tfidf/requirements.txt /tfidf/requirements.txt
WORKDIR /tfidf
RUN pip3 install -r requirements.txt
COPY /tfidf /tfidf
RUN ["chmod", "+x", "start.sh"]
EXPOSE 5000
CMD ["./start.sh"]
