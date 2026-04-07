FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY --chown=user server/requirements.txt requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
