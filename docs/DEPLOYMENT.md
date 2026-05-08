# Deployment Guide

## Quick Start

### Local Development

1. **Clone and setup:**
```bash
git clone <repo-url>
cd retail-analytics-platform
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Run the app:**
```bash
streamlit run app/app.py
```

3. **Access dashboard:**
```
http://localhost:8501
```

---

## Docker Deployment

### Using Docker

1. **Build image:**
```bash
docker build -t retail-analytics .
```

2. **Run container:**
```bash
docker run -p 8501:8501 retail-analytics
```

### Using Docker Compose

1. **Start services:**
```bash
docker-compose up -d
```

2. **Access app:**
```
http://localhost:8501
```

3. **Stop services:**
```bash
docker-compose down
```

---

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Streamlit account

### Steps

1. **Push code to GitHub**
   - Create a new repository
   - Push all files

2. **Deploy to Streamlit Cloud**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Select your repository and branch
   - Set main file to `app/app.py`
   - Click Deploy

3. **Configure secrets (if needed)**
   - Go to app settings
   - Click "Secrets"
   - Add environment variables:
   ```
   DATABASE_URL=postgresql://...
   ```

---

## AWS Deployment

### Using Elastic Beanstalk

1. **Create `.ebextensions` folder** and add nginx config:
```yaml
# .ebextensions/nginx.config
option_settings:
  aws:elasticbeanstalk:environment:proxy:staticfiles:
    /static: /app/static
```

2. **Deploy:**
```bash
eb create retail-app
eb deploy
```

---

## Azure App Service

1. **Create app:**
```bash
az webapp create --resource-group mygroup --plan myplan --name retail-app
```

2. **Configure container:**
```bash
az webapp config container set --name retail-app --docker-custom-image-name retail-analytics
```

3. **Deploy:**
```bash
git push azure main
```

---

## GCP Cloud Run

1. **Build and push image:**
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/retail-analytics
```

2. **Deploy:**
```bash
gcloud run deploy retail-analytics \
  --image gcr.io/PROJECT-ID/retail-analytics \
  --platform managed \
  --region us-central1 \
  --port 8501
```

---

## Environment Variables

Create `.env` file:
```
DATABASE_URL=postgresql://user:password@localhost:5432/retail
DEBUG=False
STREAMLIT_SERVER_PORT=8501
```

---

## Monitoring

### Logs
```bash
# Streamlit Cloud
streamlit logs

# Docker
docker logs retail-analytics-app

# Kubernetes
kubectl logs deployment/retail-analytics
```

### Performance
- Monitor CPU and memory usage
- Check response times
- Track error rates

---

## Scaling

### Horizontal Scaling
- Multiple containers/replicas
- Load balancer (nginx, HAProxy)

### Vertical Scaling
- Increase container resources
- Optimize code and data processing

---

## Troubleshooting

### Port already in use
```bash
# Find and kill process
lsof -i :8501
kill -9 <PID>
```

### Database connection issues
```bash
# Check connection string
psql $DATABASE_URL
```

### Streamlit cache issues
```bash
# Clear cache
rm -rf ~/.streamlit/cache
streamlit run app/app.py
```

---

## Best Practices

1. ✅ Use environment variables for secrets
2. ✅ Implement health checks
3. ✅ Set up monitoring and alerting
4. ✅ Use docker-compose for local dev
5. ✅ Test before deploying
6. ✅ Keep dependencies updated
7. ✅ Document deployment steps
8. ✅ Use CI/CD pipelines
