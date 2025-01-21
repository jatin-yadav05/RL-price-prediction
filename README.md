# Price Optimization with Reinforcement Learning

[![CI/CD](https://github.com/yourusername/price-optimization-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/price-optimization-rl/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rl-price-prediction.streamlit.app)


An intelligent price optimization system using reinforcement learning to maximize sales and revenue while maintaining optimal conversion rates.

## 🌟 Features

- **Dynamic Price Optimization**: Uses reinforcement learning to suggest optimal prices
- **Sales Prediction**: Incorporates predicted sales data for better decision making
- **Conversion Rate Analysis**: Considers both organic and ad conversion rates
- **Interactive Dashboard**: Beautiful Streamlit frontend for easy interaction
- **Real-time Evaluation**: Instant feedback on model performance
- **Exploration vs Exploitation**: Smart balance between trying new prices and using proven ones

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone [<your-repo-url>](https://github.com/jatin-yadav05/RL-price-prediction)
cd price-optimization-rl
```

2. **Setup (Windows)**
```bash
# Run the setup script
run_frontend.bat
```

3. **Setup (Linux/Mac)**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the application
streamlit run frontend/app.py
```

## 📊 Data Format

The system expects CSV files with the following columns:
- Report Date: Timestamp for the data point
- Product Price: Historical price point
- Organic Conversion Percentage: Natural conversion rate
- Ad Conversion Percentage: Paid advertising conversion rate
- Total Profit: Profit for that price point
- Total Sales: Units sold
- Predicted Sales: Forecasted sales for future dates

## 🤖 Model Architecture

The reinforcement learning model uses:
- **Algorithm**: Proximal Policy Optimization (PPO)
- **State Space**: Price, sales, organic conversion, ad conversion
- **Action Space**: Continuous price adjustments
- **Reward Function**: Optimizes for:
  - Higher prices above historical median
  - Maintained or increased sales
  - Improved conversion rates
  - Exploration of new price points

## 📱 Frontend Features

1. **Main Dashboard**
   - Historical price trends
   - Sales vs Price analysis
   - Real-time price recommendations

2. **Training Page**
   - Model training interface
   - Progress tracking
   - Performance metrics

3. **Evaluation Page**
   - Model performance analysis
   - Price trajectory visualization
   - Success rate metrics

## 🛠️ Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code
black src/ tests/ frontend/

# Run linting
flake8 src/ tests/ frontend/
```

## 📈 Results

The model has shown significant improvements in pricing optimization:
- Successfully pushes prices above historical median while maintaining sales
- Adapts to market conditions using conversion rates
- Provides explainable recommendations with confidence metrics

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the Streamlit team for their amazing framework
- Stable-Baselines3 team for their RL implementation

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app pointing to your forked repository
4. Select `frontend/app.py` as the main file
5. Deploy!

The app will be automatically deployed and available at: `https://[your-app-name].streamlit.app`

### Manual Deployment

For manual deployment on your own server:

```bash
# Install deployment requirements
pip install -r requirements-deploy.txt

# Run the app
streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t price-optimization-rl .

# Run the container
docker run -p 8501:8501 price-optimization-rl
``` 
