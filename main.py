import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# ایجاد اپلیکیشن FastAPI
app = FastAPI()

# بارگذاری مدل خلاصه‌سازی
summarizer = pipeline("summarization", model="sshleifer/tiny-distilbart-cnn-6-6")
# تعریف مدل ورودی درخواست
class ArticleRequest(BaseModel):
    text: str

# تعریف مسیر API برای خلاصه‌سازی متن
@app.post("/summarize")
async def summarize(article: ArticleRequest):
    summary = summarizer(article.text, max_length=150, min_length=50, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

# اجرای اولیه‌ی سرور
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
