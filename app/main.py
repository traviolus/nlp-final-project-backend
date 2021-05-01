from fastapi import FastAPI

app = FastAPI(
    title="NLP-Final-Project",
    description="Backend system for NLP 2020/2 final project => BERTong",
    version="0.1"
)

@app.get("/")
async def root():
    return {"message": "BERTong v0.1!"}