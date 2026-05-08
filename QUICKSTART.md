# Quick Start Guide

## 🚀 Get Up and Running in 5 Minutes

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional)

---

## Option 1: Local Development (Recommended)

### Step 1: Setup Environment

```bash
# Clone or download the project
cd retail-analytics-platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Dashboard

```bash
streamlit run app/app.py
```

Your dashboard will open at: **http://localhost:8501** ✨

You should see:
- 🏠 Home page with platform overview
- 📊 Data Overview section
- 💳 RFM Analysis
- 🎯 Customer Segments
- 🤖 Model Performance
- 💡 Insights & Recommendations

---

## Option 2: Docker (Fast & Easy)

### Step 1: Build Docker Image

```bash
docker build -t retail-analytics .
```

### Step 2: Run Container

```bash
docker run -p 8501:8501 retail-analytics
```

Access at: **http://localhost:8501**

---

## Option 3: Docker Compose (Full Stack)

### Step 1: Start Services

```bash
docker-compose up -d
```

This starts:
- **Streamlit App** at http://localhost:8501
- **PostgreSQL Database** at localhost:5432

### Step 2: Stop Services

```bash
docker-compose down
```

---

## 📊 Running the Full Pipeline

After installing dependencies, you can run the complete analytics pipeline:

```bash
python main.py --all
```

This will:
1. ✅ Explore raw data sources
2. ✅ Run RFM analysis pipeline
3. ✅ Perform customer clustering
4. ✅ Train ML models
5. ✅ Generate visualizations

---

## 📁 Project Structure Quick Reference

```
📦 retail-analytics-platform/
 ├── 📄 app/                    # Streamlit dashboard
 ├── 📄 raw_data/               # Original datasets
 ├── 📄 clean_data/             # Cleaned datasets
 ├── 📄 modeling/               # ML models & analysis
 ├── 📄 docs/                   # Documentation
 ├── 📄 requirements.txt        # Dependencies
 ├── 📄 Dockerfile              # Docker setup
 ├── 📄 docker-compose.yml      # Full stack setup
 └── 📄 README.md               # Full documentation
```

---

## 🎯 Key Features to Explore

### Dashboard Pages

1. **Home** 🏠
   - Platform overview
   - Key metrics
   - Getting started guide

2. **Data Overview** 📊
   - Dataset statistics
   - Data quality metrics
   - Source comparison

3. **RFM Analysis** 💳
   - Customer value metrics
   - Segment distribution
   - Behavioral insights

4. **Customer Segments** 🎯
   - K-means clustering results
   - Segment characteristics
   - Interactive exploration

5. **Model Performance** 🤖
   - ML model metrics
   - Model comparison
   - Performance charts

6. **Insights** 💡
   - Key findings
   - Business recommendations
   - Action items

---

## ⚙️ Configuration

### Modify Dashboard Settings

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"           # Change primary color
backgroundColor = "#f8f9fa"        # Change background
font = "sans serif"                # Change font

[server]
port = 8501                        # Change port number
```

### Environment Variables

Create `.env` file:

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/retail
DEBUG=False
STREAMLIT_SERVER_PORT=8501
```

---

## 🔧 Troubleshooting

### Issue: Port 8501 already in use

```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>
```

### Issue: Dependencies not installing

```bash
# Upgrade pip first
pip install --upgrade pip

# Try installing again
pip install -r requirements.txt
```

### Issue: Streamlit cache errors

```bash
# Clear cache
rm -rf ~/.streamlit/cache

# Run again
streamlit run app/app.py
```

### Issue: Database connection fails

```bash
# Check PostgreSQL is running
psql --version

# Or use without database (demo mode)
# App will work with sample data
```

---

## 📚 Next Steps

1. **Explore the Dashboard**
   - Navigate through all pages
   - Check data quality metrics
   - Review model performance

2. **Read Documentation**
   - [docs/PIPELINE.md](docs/PIPELINE.md) - Data pipeline details
   - [docs/MODELS.md](docs/MODELS.md) - ML model documentation
   - [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment guide

3. **Customize the Project**
   - Modify RFM thresholds in `modeling/rfm_analysis.py`
   - Add new visualizations in `app/app.py`
   - Adjust ML parameters in `modeling/supervised_models.py`

4. **Deploy to Production**
   - [Streamlit Cloud](#streamlit-cloud-deployment) - Easiest option
   - [Docker](#option-2-docker-fast--easy) - Flexible deployment
   - [AWS/Azure/GCP](#deployment-options) - Enterprise scale

---

## 🚀 Deployment

### Quick Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select repository and `app/app.py` as main file
5. Click Deploy ✨

**Free tier available!**

---

## 📞 Support & Resources

- 📖 Full [README.md](README.md) for comprehensive documentation
- 📚 [Deployment Guide](docs/DEPLOYMENT.md) for production setup
- 🤖 [Models Documentation](docs/MODELS.md) for ML details
- 🔄 [Pipeline Guide](docs/PIPELINE.md) for ETL information

---

## 💡 Tips

✅ Start with the dashboard to understand data and results
✅ Check `modeling/pipeline_data/` for saved models and data
✅ Review visualizations in `modeling/visualize_model/` directory
✅ Use `docker-compose up` for complete local development environment
✅ Deploy to Streamlit Cloud for free public hosting

---

## 🎉 You're Ready!

Your analytics platform is now running. Start exploring! 🚀

```bash
streamlit run app/app.py
```

**Visit: http://localhost:8501**

---

Feel free to customize and extend based on your needs!
