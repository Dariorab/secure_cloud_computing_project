FROM python:3.10.9-slim
WORKDIR /app
# Copy requirements
COPY requirements_app.txt ./requirements_app.txt
# Install dependencies
RUN pip3 install -r requirements_app.txt
# Expose port
EXPOSE 8501
COPY . /app
# Create an entry point to make the image executable
ENTRYPOINT ["streamlit", "run"]
# Run the application:
CMD ["app.py"]