"""Launch the local xlog site: python run.py -> http://127.0.0.1:8321"""
import uvicorn

if __name__ == "__main__":
    # No reload: a code edit mid-job restarts the server and strands the job.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8321)
