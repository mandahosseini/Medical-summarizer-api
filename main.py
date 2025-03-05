from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# بارگذاری مدل خلاصه‌سازی
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class ArticleRequest(BaseModel):
    text: str

@app.post("/summarize/")
def summarize(article: ArticleRequest):
    summary = summarizer(article.text, max_length=150, min_length=50, do_sample=False)
    return {"summary": summary[0]["summary_text"]}