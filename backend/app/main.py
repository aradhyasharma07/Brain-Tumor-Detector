from fastapi import FastAPI

app = FastAPI(title="Brain Tumor Detection API")


@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}