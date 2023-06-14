FROM nginx
COPY docs/_build/html /usr/share/nginx/html/
RUN ls -la /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf