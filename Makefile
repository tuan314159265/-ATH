# Common commands for development and deployment

.PHONY: help install run test docker-build docker-run clean docs

help:
	@echo "🛍️ Retail Analytics Platform - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install         - Install dependencies"
	@echo "  make install-dev     - Install dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run            - Run Streamlit app"
	@echo "  make pipeline       - Run full analytics pipeline"
	@echo "  make clean          - Clean up temporary files"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make docker-stop    - Stop Docker container"
	@echo "  make compose-up     - Start docker-compose stack"
	@echo "  make compose-down   - Stop docker-compose stack"
	@echo ""
	@echo "Quality:"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Lint with flake8"
	@echo "  make test           - Run tests"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - View documentation"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

install-dev:
	@echo "📦 Installing dev dependencies..."
	pip install -r requirements.txt
	pip install pytest black flake8 pylint jupyter
	@echo "✅ Dev dependencies installed"

run:
	@echo "🚀 Starting Streamlit app..."
	streamlit run app/app.py

pipeline:
	@echo "⚙️ Running analytics pipeline..."
	python main.py --all

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .streamlit/cache
	rm -rf build dist *.egg-info
	@echo "✅ Cleanup complete"

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t retail-analytics .
	@echo "✅ Docker image built"

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8501:8501 --name retail-app retail-analytics
	@echo "✅ Container running at http://localhost:8501"

docker-stop:
	@echo "⛔ Stopping Docker container..."
	docker stop retail-app
	docker rm retail-app
	@echo "✅ Container stopped"

compose-up:
	@echo "🐳 Starting docker-compose stack..."
	docker-compose up -d
	@echo "✅ Stack running"
	@echo "   App: http://localhost:8501"
	@echo "   DB:  localhost:5432"

compose-down:
	@echo "⛔ Stopping docker-compose stack..."
	docker-compose down
	@echo "✅ Stack stopped"

format:
	@echo "🎨 Formatting code with black..."
	black app/ modeling/ cleaning/ integration/ tranformation_data/ merge_data/ load/ main.py
	@echo "✅ Code formatted"

lint:
	@echo "🔍 Linting with flake8..."
	flake8 app/ modeling/ cleaning/ integration/ tranformation_data/ merge_data/ load/ main.py --max-line-length=100
	@echo "✅ Linting complete"

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v
	@echo "✅ Tests complete"

docs:
	@echo "📚 Documentation:"
	@echo "   - README.md (Main documentation)"
	@echo "   - QUICKSTART.md (Quick setup guide)"
	@echo "   - PORTFOLIO.md (CV/Portfolio summary)"
	@echo "   - docs/PIPELINE.md (Pipeline documentation)"
	@echo "   - docs/MODELS.md (ML models documentation)"
	@echo "   - docs/DEPLOYMENT.md (Deployment guide)"
