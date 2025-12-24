
# Dockerfile for production mode
#
# ---------------------------------------------------------
#
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_DIR=/opt/poetry \
    BACKEND_DIR=/home/ai

#
# ---------------------------------------------------------
#
FROM base AS poetry
WORKDIR $POETRY_DIR
COPY ./poetry.lock ./pyproject.toml $POETRY_DIR/
RUN pip install --timeout=100 poetry -U \
    && pip install --timeout=100 poetry-plugin-export -U \
    && poetry export --without-hashes --without dev -f requirements.txt -o requirements.txt

#
# ---------------------------------------------------------
#
FROM base AS deps
WORKDIR $BACKEND_DIR
COPY ./deploy/pre-install.sh $BACKEND_DIR/pre-install.sh
RUN chmod a+x $BACKEND_DIR/pre-install.sh && $BACKEND_DIR/pre-install.sh
COPY --from=poetry $POETRY_DIR/requirements.txt $BACKEND_DIR/
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

#
# ---------------------------------------------------------
#
FROM base AS runner
USER root
WORKDIR $BACKEND_DIR
COPY ./deploy/pre-install.sh $BACKEND_DIR/pre-install.sh
RUN chmod a+x $BACKEND_DIR/pre-install.sh && $BACKEND_DIR/pre-install.sh
COPY --from=deps /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=deps /usr/local/bin/ /usr/local/bin/
COPY deploy/train.sh ./
COPY . $BACKEND_DIR/
RUN chmod a+x $BACKEND_DIR/train.sh
