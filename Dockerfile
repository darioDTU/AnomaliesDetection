FROM python:3.12.5-bullseye

WORKDIR /app

RUN pip install numpy==2.1.2
RUN pip install pandas==2.2.3
RUN pip install matplotlib==3.10.3
RUN pip install scipy==1.15.3
RUN pip install xarray==2025.6.1
RUN pip install bokeh==3.7.3
RUN pip install argopy==1.2.0
RUN pip install uvicorn==0.34.3
RUN pip install fastapi==0.115.12

COPY . /app

CMD ["uvicorn", "ada_api:app", "--host", "0.0.0.0", "--port", "8000"]