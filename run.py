"""Launch the local xlog site: python run.py -> http://127.0.0.1:8321"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8321, reload=True)
