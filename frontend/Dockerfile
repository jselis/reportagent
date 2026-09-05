FROM node:20-slim AS base

WORKDIR /code

COPY package.json .
RUN npm install

# --- dev: what docker-compose.yml uses locally (Vite dev server, hot reload) ---
FROM base AS dev

COPY . .

CMD ["npm", "run", "dev"]

# --- build: produces the static production bundle ---
FROM base AS build

ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

COPY . .
RUN npm run build

# --- prod: serves the static bundle, used for Railway/Render deploys ---
FROM base AS prod

RUN npm install -g serve

COPY --from=build /code/dist ./dist

CMD exec serve -s dist -l $PORT
