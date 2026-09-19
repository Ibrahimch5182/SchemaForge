# OPTIONAL: static frontend for local Compose validation only. The public frontend is deployed to
# Vercel (frontend/vercel.json); this image is not part of the cloud runbook.
#   docker build -f docker/frontend.Dockerfile --build-arg VITE_API_BASE_URL=<backend url> .
FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY docker/frontend-nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
