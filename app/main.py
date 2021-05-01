from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import default_router

app = FastAPI(
    title="NLP-Final-Project",
    description="Backend system for NLP 2020/2 final project => BERTong",
    version="0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(default_router)

@app.get("/")
async def root():
    return {"message": "BERTong v0.1!"}