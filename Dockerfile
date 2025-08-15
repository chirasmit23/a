FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose the port
EXPOSE 7860

# Run the app (change this to your main file)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
