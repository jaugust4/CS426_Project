# Pokémon Legendary Status Predictor

## Project Overview
This project implements a machine learning-based system to predict whether a Pokémon is legendary based on its characteristics and statistics. The system employs a Random Forest Classifier trained on comprehensive Pokémon data, with a focus on handling class imbalance and optimizing model performance.

## Technical Implementation

### Data Processing Pipeline
1. **Data Collection and Preprocessing**
   - Utilizes a comprehensive Pokémon dataset containing base stats, physical characteristics, and type information
   - Implements data cleaning and feature engineering
   - Handles missing values using SimpleImputer with most frequent strategy

2. **Feature Engineering**
   - Numerical features: HP, Attack, Defense, Special Attack, Special Defense, Speed, Height, Weight, Generation
   - Categorical features: Primary and Secondary Types (one-hot encoded)
   - Target variable: Binary classification (Legendary vs Non-Legendary)

3. **Model Architecture**
   - Random Forest Classifier with optimized hyperparameters
   - Class weights to address imbalanced data
   - SMOTE (Synthetic Minority Over-sampling Technique) for data augmentation to help balance training dataset
   - Stratified sampling for training/test split

### System Components

#### 1. Data Analysis Module (`pokemon_models.py`)
- Implements data structures for Pokémon statistics
- Provides methods for statistical analysis
- Generates visualizations for data exploration

#### 2. Machine Learning Module (`pokemon_ml.py`)
- Implements the core prediction system
- Handles data preprocessing and feature engineering
- Trains and evaluates the Random Forest model
- Provides model interpretability through feature importance analysis
- Implements prediction interface for new Pokémon

#### 3. Dashboard Interface (`dashboard.py`)
- Interactive Streamlit-based web interface
- Real-time model evaluation and visualization
- Interactive prediction interface
- Comprehensive performance metrics display

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements.txt):
  - pandas: Data manipulation and analysis
  - scikit-learn: Machine learning implementation
  - streamlit: Web interface framework
  - plotly: Interactive visualizations
  - imbalanced-learn: Handling class imbalance

### Installation Steps
1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
```bash
streamlit run scripts/dashboard.py
```

## Technical Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 10
  - min_samples_leaf: 5
  - max_features: 'sqrt'
  - class_weight: balanced

### Performance Optimization
- SMOTE implementation for class balancing
- Stratified sampling for representative test sets
- Feature importance analysis for model interpretability
- Regularization techniques to prevent overfitting

## References
- Pokémon Dataset: Publicly available Pokémon statistics database
- Scikit-learn Documentation: Machine learning implementation reference
- Streamlit Documentation: Web interface framework reference
- Imbalanced-learn Documentation: Class imbalance handling reference