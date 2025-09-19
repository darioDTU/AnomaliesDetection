# 1. Use an official Node.js image as the build environment
FROM node:20 AS build

# 2. Set working directory
WORKDIR /app

# 3. Copy package.json and package-lock.json (or yarn.lock)
COPY package*.json ./

# 4. Install dependencies
RUN npm install

# 5. Copy the rest of the app source code
COPY . .

# 6. Build the React app for production
RUN npm run build

# 7. Use a lightweight web server to serve the build (e.g., nginx)
FROM nginx:stable-alpine

# 8. Copy the build output to nginx's html directory
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom nginx.conf into container
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# 9. Expose port 80
EXPOSE 80

# 10. Start nginx
CMD ["nginx", "-g", "daemon off;"]