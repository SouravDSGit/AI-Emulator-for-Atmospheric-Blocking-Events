# AI Emulator for Atmospheric Blocking Events

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/blocking-emulator/blob/main/blocking_emulator.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An educational, fully-runnable Google Colab notebook** demonstrating how to build a deep learning emulator for predicting atmospheric blocking events—major drivers of extreme weather like heatwaves, droughts, and cold snaps.

## 📚 What You'll Learn

This notebook teaches you how to:

- ✅ Build neural networks for atmospheric science applications
- ✅ Create physics-informed AI models that learn climate patterns
- ✅ Engineer temporal features for time series prediction
- ✅ Evaluate classification and regression models simultaneously
- ✅ Interpret AI predictions through feature importance analysis
- ✅ Achieve 2.5 million times speedup over traditional physics models

**Perfect for:** Climate scientists learning ML, data scientists interested in Earth science, students studying AI applications in physics-based systems.

## 🌍 Background: Why Atmospheric Blocking Matters

### What is Atmospheric Blocking?

Atmospheric blocking happens when a large high-pressure system gets "stuck" in the atmosphere for days or weeks, blocking the normal flow of weather systems. Think of it like a traffic jam in the atmosphere.

**Real-world impacts:**
- 🔥 **2003 European Heatwave**: Blocking event caused 70,000+ deaths
- ❄️ **2010 Russian Heatwave**: Persistent blocking led to wildfires and crop failures
- 🌡️ **2021 Pacific Northwest Heat Dome**: Record-breaking temperatures from blocking

### The Problem This Notebook Solves

Traditional physics-based atmospheric models (like GFS, ECMWF) are:
- ⏱️ **Slow**: Take hours on supercomputers
- 💰 **Expensive**: Require massive computational resources
- 🔒 **Limited**: Can't run thousands of scenarios for uncertainty quantification

**Our solution:** Train a neural network to emulate blocking predictions in **milliseconds** while maintaining high accuracy.

## 🚀 Getting Started

### Option 1: Google Colab (Recommended - No Setup Required!)

1. Click the "Open in Colab" badge at the top of this README
2. Click **Runtime → Run All** in the Colab menu
3. Wait ~5-10 minutes for complete execution
4. Explore the results!

**That's it!** No installation, no configuration, no downloads. Everything runs in your browser.

### Option 2: Local Jupyter Notebook

If you prefer to run locally:

```bash
# Clone this repository
git clone https://github.com/SouravDSGit/AI-Emulator-for-Atmospheric-Blocking-Events.git
cd AI-Emulator-for-Atmospheric-Blocking-Events

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook AI_Emulator_for_Atmospheric_Blocking_Events.ipynb
```

**Requirements:**
- Python 3.8 or higher
- TensorFlow 2.x
- Standard scientific Python stack (NumPy, Pandas, Matplotlib, scikit-learn)

## 📖 What This Notebook Does

### Complete Workflow (13 Sections)

#### **Section 1-2: Setup & Data Generation**
- Automatically installs all required packages
- Generates 10 years of synthetic atmospheric data that mimics real ERA5 reanalysis
- Creates realistic blocking events with proper persistence patterns

**Why synthetic data?** 
- Real ERA5 data requires account registration (24-48 hour approval)
- Large download sizes (GBs)
- This synthetic data has the same statistical properties as real data
- Perfect for learning and demonstration

#### **Section 3-4: Exploratory Data Analysis**
- Visualizes atmospheric variables (temperature, pressure, geopotential height)
- Identifies blocking events in the data
- Shows seasonal patterns and correlations
- Creates publication-quality figures automatically

**You'll see:**
- Time series of atmospheric conditions
- Distribution of blocking vs non-blocking days
- Seasonal blocking frequency (more common in winter!)
- How variables correlate with blocking

#### **Section 5-6: Feature Engineering & Model Building**
- Creates lagged features (uses past 3 days to predict 5 days ahead)
- Engineers temporal context that captures atmospheric persistence
- Builds a multi-task neural network (predicts both blocking occurrence AND magnitude)
- Explains each layer and why it's designed that way

**Neural Network Architecture:**
```
Input: 16 features (4 variables × 4 time steps)
  ↓
Hidden Layer 1: 128 neurons + ReLU + Dropout
  ↓
Hidden Layer 2: 64 neurons + ReLU + Dropout
  ↓
Hidden Layer 3: 32 neurons + ReLU
  ↓
Output 1: Blocking probability (0-100%)
Output 2: Z500 prediction (geopotential height in meters)
```

#### **Section 7-8: Training & Evaluation**
- Trains the model with proper train/validation/test splits
- Uses early stopping to prevent overfitting
- Implements learning rate scheduling for better convergence
- Evaluates on held-out test data (the model never sees this during training!)

**Training features:**
- Multi-task learning (learns two related tasks simultaneously)
- Automatic checkpointing (saves best model)
- Progress monitoring with validation metrics

#### **Section 9-10: Results & Visualization**
- Creates comprehensive performance visualizations
- Shows prediction accuracy vs actual values
- Generates ROC curves for classification performance
- Displays time series comparisons

**You'll get:**
- 3 high-resolution PNG figures saved automatically
- Detailed performance metrics
- Comparison plots showing model vs reality

#### **Section 11-12: Feature Importance & Performance Analysis**
- Analyzes which atmospheric variables matter most
- Calculates computational speedup vs physics models
- Provides physical interpretation of results

**Key insight:** The model learns that current Z500 (geopotential height) is most important—exactly what atmospheric physics tells us!

#### **Section 13: Summary Report**
- Generates a complete text summary of the project
- Lists all metrics and findings
- Provides context for applications

### 📊 What Results You'll Get

After running the notebook, you'll have:

**3 Publication-Quality Figures:**
1. `atmospheric_analysis.png` - Data exploration and patterns
2. `emulator_performance.png` - Model predictions vs actual values
3. `feature_importance.png` - Which variables matter most

**1 Summary Report:**
- `project_summary.txt` - Complete analysis writeup

**Model Performance (Typical Results):**
- **Blocking Detection Accuracy**: ~85-90%
- **ROC-AUC Score**: ~0.85
- **Z500 Prediction R²**: ~0.90+
- **Inference Speed**: ~0.4 milliseconds per prediction
- **Speedup vs Physics Models**: ~2,500,000× faster!

## 🔬 Technical Deep Dive

### Data Structure

Each sample contains:
- **Z500**: 500mb geopotential height (primary blocking indicator)
- **SLP**: Sea level pressure (surface conditions)
- **T850**: 850mb temperature (lower atmosphere)
- **T500**: 500mb temperature (mid-troposphere)

For each variable, we use 4 time steps: t-3, t-2, t-1, t-0 (current day)

**Total input features**: 4 variables × 4 time steps = 16 features

### Why These Variables?

**Z500 (Geopotential Height at 500mb):**
- The height of the 500 millibar pressure surface (~5.5 km altitude)
- Blocking = anomalously high Z500 (ridge pattern)
- This is THE variable meteorologists use to identify blocking

**SLP (Sea Level Pressure):**
- Surface manifestation of blocking
- High SLP at surface often accompanies blocking aloft

**T850 & T500 (Temperatures):**
- Thermodynamic component of blocking
- Warm anomalies often associated with blocking ridges

### Model Design Choices

**Why Multi-Task Learning?**
- Predicting both blocking (classification) and Z500 value (regression)
- Both tasks share atmospheric physics knowledge
- Improves generalization and prevents overfitting
- More efficient than training two separate models

**Why Dropout?**
- Prevents the model from memorizing training data
- Forces it to learn robust atmospheric patterns
- Improves performance on new, unseen data

**Why 3 Hidden Layers?**
- Layer 1: Learns basic patterns
- Layer 2: Learns combinations of patterns
- Layer 3: Learns complex atmospheric dynamics
- More layers = can learn more complex relationships

### Evaluation Strategy

**Train/Validation/Test Split:**
- 70% training (model learns from this)
- 15% validation (tunes hyperparameters)
- 15% test (final evaluation - model never sees this!)

**Why preserve temporal order?**
- Climate data has autocorrelation (adjacent days are similar)
- Random shuffling would leak information
- We maintain temporal sequence for realistic evaluation

## 🎓 Educational Value

### For Climate Scientists
- Learn how to apply deep learning to atmospheric data
- See how AI can complement traditional physics models
- Understand feature engineering for time series
- Get started with TensorFlow/Keras

### For Data Scientists
- Apply ML to a real-world physics problem
- Learn domain-specific feature engineering
- See how to interpret model predictions physically
- Understand multi-task learning applications

### For Students
- Complete end-to-end ML project
- Well-documented, educational code
- Real scientific application
- Can adapt for thesis/coursework

## 💡 Extension Ideas

Want to take this further? Try:

1. **Use Real Data**: Replace synthetic data with actual ERA5 reanalysis from Copernicus Climate Data Store
2. **Multi-Location**: Extend from single point to multiple grid points
3. **Ensemble Predictions**: Train multiple models and combine predictions
4. **Longer Forecasts**: Predict 10-15 days ahead instead of 5
5. **Add More Variables**: Include wind speed, humidity, etc.
6. **Uncertainty Quantification**: Add Bayesian layers or Monte Carlo dropout
7. **Spatial Models**: Use CNNs to capture 2D spatial patterns
8. **Compare Models**: Try LSTM, Transformer, or other architectures

## 🤝 Contributing

This is an educational resource! Contributions welcome:

- 🐛 Found a bug? Open an issue
- 💡 Have an improvement? Submit a pull request
- 📚 Want to add documentation? Please do!
- 🎓 Using this for teaching? Let us know!

## 📄 License

MIT License - feel free to use this for learning, teaching, or research!

## 🙏 Acknowledgments & References

This notebook was created as an educational resource combining concepts from:

### Scientific Background
- **ERA5 Reanalysis**: Hersbach et al. (2020), "The ERA5 global reanalysis", *Quarterly Journal of the Royal Meteorological Society*
- **Blocking Dynamics**: Rex (1950), "Blocking action in the middle troposphere and its effect upon regional climate", *Tellus*
- **Extreme Events**: Barriopedro et al. (2011), "The hot summer of 2010: redrawing the temperature record map of Europe", *Science*

### Machine Learning Methods
- **Deep Learning Framework**: TensorFlow/Keras documentation
- **Multi-Task Learning**: Caruana (1997), "Multitask Learning", *Machine Learning*
- **Feature Importance**: Gradient-based attribution methods

### Data & Methodology
- **Synthetic Data Generation**: Statistical properties based on ERA5 characteristics (mean, variance, seasonal cycle, blocking frequency)
- **Blocking Detection**: Based on geopotential height anomalies (Z500 > 95th percentile with persistence)
- **Model Architecture**: Standard feedforward neural network with dropout regularization

### Inspiration
- Climate informatics community
- WeatherBench (Rasp et al., 2020)
- ClimateBench (Watson-Parris et al., 2022)
- AI for Earth system modeling workshops

### Tools & Libraries
- **TensorFlow**: Abadi et al. (2015), "TensorFlow: Large-scale machine learning on heterogeneous systems"
- **NumPy**: Harris et al. (2020), "Array programming with NumPy", *Nature*
- **Pandas**: McKinney (2010), "Data structures for statistical computing in Python"
- **Scikit-learn**: Pedregosa et al. (2011), "Scikit-learn: Machine learning in Python", *JMLR*
- **Matplotlib**: Hunter (2007), "Matplotlib: A 2D graphics environment", *Computing in Science & Engineering*

## 📧 Contact & Questions

- **Issues**: Use GitHub Issues for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: [soumukhcivil@gmail.com] for other inquiries

## 🌟 Citation

If you use this notebook for research or teaching, please cite:

```bibtex
@software{blocking_emulator,
  author = {Sourav Mukherjee},
  title = {AI Emulator for Atmospheric Blocking Events},
  year = {2025},
  url = {https://github.com/SouravDSGit/AI-Emulator-for-Atmospheric-Blocking-Events}
}
```

---

**Happy Learning! 🚀**

*Remember: This is a demonstration project using synthetic data. For operational forecasting or research publications, use real atmospheric reanalysis data (ERA5, MERRA-2, etc.) and validate against observations.*
