FROM node:20-slim

WORKDIR /code

COPY package.json .
RUN npm install

COPY . .

CMD ["npm", "run", "dev"]
