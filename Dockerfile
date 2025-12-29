# 1. Use a lightweight Linux with Python 3.10 pre-installed
FROM python:3.10-slim

# 2. Create a folder named 'app' inside that Linux machine
WORKDIR /app

# 3. Copy all files from your laptop's current folder into the 'app' folder
COPY . .

# 4. Install the needed libraries
RUN pip install --no-cache-dir \
    google-cloud-storage \
    google-cloud-firestore \
    google-cloud-aiplatform \
    pandas \
    numpy \
    tensorflow \
    scikit-learn \
    keras-tuner

# 5. Set the entry point to your application
ENTRYPOINT ["python", "-m", "train.task"]