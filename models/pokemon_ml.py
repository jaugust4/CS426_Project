import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, average_precision_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# This class helps us predict if a Pokemon is legendary
class PokemonML:
    def __init__(self, df):
        # Store the Pokemon data
        self.df = df
        # This will store our trained model
        self.model = None
        # This will store which features are most important
        self.feature_importance = None
        # These will store our training and testing data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # This helps fill in missing data
        self.imputer = SimpleImputer(strategy='most_frequent')
        # This helps balance our data
        self.smote = SMOTE(random_state=42)
        # This tells us if the model is trained
        self.is_trained = False
        
    def prepare_data(self):
        # These are the features we'll use to predict if a Pokemon is legendary
        features = [
            'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
            'height_m', 'weight_kg', 'generation'
        ]
        
        # Convert Pokemon types to numbers
        type_dummies = pd.get_dummies(self.df[['type1', 'type2']].fillna('None'))
        
        # Put all our features together
        X = pd.concat([
            self.df[features],
            type_dummies
        ], axis=1)
        
        # Fill in any missing data
        X = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns
        )
        
        # This is what we're trying to predict (1 = legendary, 0 = not legendary)
        y = self.df['is_legendary']
        
        # Split our data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Make more legendary Pokemon data to balance our dataset
        self.X_train, self.y_train = self.smote.fit_resample(self.X_train, self.y_train)
        
        return X, y
    
    def train_model(self):
        # Get our data ready
        X, y = self.prepare_data()
        
        # Give more importance to legendary Pokemon since there aren't many of them
        class_weights = {
            0: 1.0,
            1: len(self.df[self.df['is_legendary'] == 0]) / len(self.df[self.df['is_legendary'] == 1])
        }
        
        # Create our model (Random Forest is like a bunch of decision trees working together)
        self.model = RandomForestClassifier(
            n_estimators=100,        # How many trees to use
            max_depth=10,            # How deep each tree can go
            min_samples_split=10,    # Need at least 10 Pokemon to make a split
            min_samples_leaf=5,      # Need at least 5 Pokemon in each leaf
            max_features='sqrt',     # Only look at some features at each split
            class_weight=class_weights,
            random_state=42
        )
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Figure out which features are most important
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Mark that our model is trained
        self.is_trained = True
        
        return self.model
    
    def evaluate_model(self):
        # Make sure we trained the model first
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions from our model
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate how well our model did
        cm = confusion_matrix(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        return None, cm, precision, recall, avg_precision
    
    def get_feature_importance_plot(self):
        # Make sure we have feature importance data
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet")
            
        # Create a bar chart of the most important features
        fig = px.bar(
            self.feature_importance.head(25),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 25 Most Important Features for Legendary Status',
            color='importance',
            color_continuous_scale='Viridis',
            height=800  # Increase height to accommodate more features
        )
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_title='Feature Importance Score',
            yaxis_title='Feature',
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=150, r=50, t=50, b=50),  # Adjust margins for better fit
            showlegend=False
        )
        
        # Add value labels to bars
        fig.update_traces(
            texttemplate='%{x:.3f}',
            textposition='outside',
            textfont_size=10
        )
        
        return fig
    
    def get_confusion_matrix_plot(self, cm=None):
        # If no confusion matrix is provided, calculate one
        if cm is None:
            cm = self.get_confusion_matrix(self.X_test, self.y_test)
        
        # Create a heatmap of the confusion matrix
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Non-Legendary', 'Legendary'],
            y=['Non-Legendary', 'Legendary'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        
        # Make the plot look nice
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=500
        )
        
        return fig

    def get_confusion_matrix(self, X, y):
        # Make sure we trained the model first
        if not self.is_trained:
            raise ValueError("Model must be trained before getting confusion matrix")
        
        # Get predictions and create confusion matrix
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred)
        return cm
    
    def predict_legendary_probability(self, pokemon_data):
        # Make sure we trained the model first
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Convert the input data to the right format
        input_df = pd.DataFrame([pokemon_data])
        
        # Convert Pokemon types to numbers
        type_dummies = pd.get_dummies(input_df[['type1', 'type2']].fillna('None'))
        
        # Get all the features we need
        features = [
            'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
            'height_m', 'weight_kg', 'generation'
        ]
        
        # Put all the features together
        X = pd.concat([
            input_df[features],
            type_dummies
        ], axis=1)
        
        # Make sure we have all the columns we need
        for col in self.X_train.columns:
            if col not in X.columns:
                X[col] = 0
        
        # Put the columns in the right order
        X = X[self.X_train.columns]
        
        # Fill in any missing data
        X = pd.DataFrame(
            self.imputer.transform(X),
            columns=X.columns
        )
        
        # Get the probability that this Pokemon is legendary
        prob = self.model.predict_proba(X)[0][1]
        
        return prob 