import sys
from pathlib import Path

# Add the project folder to Python's path so it can find our files
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the tools we need
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models.pokemon_models import Pokemon, PokemonAnalysis
from models.pokemon_ml import PokemonML

# Set up the page
st.set_page_config(
    page_title="Pokémon Data Analysis Dashboard",
    page_icon="🔮",
    layout="wide"
)

# Load our Pokemon data
@st.cache_data
def load_data():
    # Find the Pokemon data file
    csv_path = project_root / 'pokemon.csv'
    # Read the data
    df = pd.read_csv(csv_path)
    # Convert each row to a Pokemon object
    pokemon_list = [Pokemon.from_dataframe(row) for _, row in df.iterrows()]
    # Create our analysis object
    return PokemonAnalysis(pokemon_list), df

# Load the data
analysis, df = load_data()

# Create the sidebar for navigation
st.sidebar.title("Pokémon Analysis Dashboard")
page = st.sidebar.radio(
    "Select Page",
    ["Data Analysis", "Machine Learning"]
)

# Data Analysis Page
if page == "Data Analysis":
    st.title("Pokémon Data Analysis")
    
    # Show some basic stats
    st.header("Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pokémon", len(analysis.pokemon_data))
    with col2:
        # Count how many legendary Pokemon there are
        legendary_count = sum(1 for p in analysis.pokemon_data if p.is_legendary)
        st.metric("Legendary Pokémon", legendary_count)
    with col3:
        # Calculate average total stats
        avg_stats = sum(p.stats.total_stats for p in analysis.pokemon_data) / len(analysis.pokemon_data)
        st.metric("Average Total Stats", f"{avg_stats:.1f}")
    
    # Show a pie chart of Pokemon types
    st.subheader("Type Distribution")
    type_dist = analysis.get_type_distribution()
    fig = px.pie(
        values=list(type_dist.values()),
        names=list(type_dist.keys()),
        title="Pokémon Type Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare stats between legendary and non-legendary Pokemon
    st.subheader("Stat Analysis by Legendary Status")
    
    # Get all the stats for each Pokemon
    stats_data = []
    for pokemon in analysis.pokemon_data:
        stats_data.append({
            'name': pokemon.name,
            'is_legendary': 'Legendary' if pokemon.is_legendary else 'Non-Legendary',
            'hp': pokemon.stats.hp,
            'attack': pokemon.stats.attack,
            'defense': pokemon.stats.defense,
            'sp_attack': pokemon.stats.sp_attack,
            'sp_defense': pokemon.stats.sp_defense,
            'speed': pokemon.stats.speed,
            'total_stats': pokemon.stats.total_stats
        })
    stats_df = pd.DataFrame(stats_data)
    
    # Make box plots for each stat
    stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'total_stats']
    for stat in stats:
        fig = px.box(
            stats_df,
            x='is_legendary',
            y=stat,
            title=f'{stat.upper()} Distribution by Legendary Status',
            color='is_legendary',
            color_discrete_map={'Legendary': 'gold', 'Non-Legendary': 'blue'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Show Pokemon counts by generation
    st.subheader("Generation Analysis")
    gen_stats = analysis.get_generation_stats()
    gen_df = pd.DataFrame(gen_stats).T
    
    # Make a bar chart showing legendary vs non-legendary by generation
    gen_data = []
    for gen in gen_df.index:
        gen_data.append({
            'Generation': gen,
            'Count': gen_df.loc[gen, 'count'] - gen_df.loc[gen, 'legendary_count'],
            'Type': 'Non-Legendary'
        })
        gen_data.append({
            'Generation': gen,
            'Count': gen_df.loc[gen, 'legendary_count'],
            'Type': 'Legendary'
        })
    gen_plot_df = pd.DataFrame(gen_data)
    
    fig = px.bar(
        gen_plot_df,
        x='Generation',
        y='Count',
        color='Type',
        title='Pokémon Count by Generation and Legendary Status',
        color_discrete_map={'Legendary': 'gold', 'Non-Legendary': 'blue'},
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

# Machine Learning Page
else:
    st.title("Pokémon Legendary Status Prediction")
    
    # Create and train our model
    ml_model = PokemonML(df)
    ml_model.train_model()
    
    # Show how well our model is doing
    st.header("Model Evaluation")
    
    # Show training and testing results
    st.subheader("Training and Testing Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Set Statistics**")
        train_stats = {
            'Total Samples': len(ml_model.y_train),
            'Non-Legendary': sum(ml_model.y_train == 0),
            'Legendary': sum(ml_model.y_train == 1),
            'Legendary Ratio': f"{sum(ml_model.y_train == 1) / len(ml_model.y_train):.2%}"
        }
        st.write(pd.DataFrame([train_stats]).T)
    
    with col2:
        st.write("**Testing Set Statistics**")
        test_stats = {
            'Total Samples': len(ml_model.y_test),
            'Non-Legendary': sum(ml_model.y_test == 0),
            'Legendary': sum(ml_model.y_test == 1),
            'Legendary Ratio': f"{sum(ml_model.y_test == 1) / len(ml_model.y_test):.2%}"
        }
        st.write(pd.DataFrame([test_stats]).T)
    
    # Show confusion matrices
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Set Confusion Matrix**")
        train_cm = ml_model.get_confusion_matrix(ml_model.X_train, ml_model.y_train)
        fig = ml_model.get_confusion_matrix_plot(train_cm)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Testing Set Confusion Matrix**")
        test_cm = ml_model.get_confusion_matrix(ml_model.X_test, ml_model.y_test)
        fig = ml_model.get_confusion_matrix_plot(test_cm)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show which features are most important
    st.subheader("Feature Importance")
    fig = ml_model.get_feature_importance_plot()
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the model updates
    st.subheader("Model Analysis")
    st.markdown("""
    The model showed signs of overfitting to the training data, as evidenced by:
    
    1. **Training vs Testing Performance**: The confusion matrices showed significantly better performance on the training set compared to the testing set
    2. **High Training Accuracy**: The model achieved near-perfect accuracy on the training data
    3. **Lower Testing Performance**: The model's performance dropped when applied to unseen data
    
    To address this, we've implemented several techniques:
    
    1. **Class Balancing**: Using SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic legendary Pokémon samples
    2. **Class Weights**: Assigning higher weights to the minority class (legendary Pokémon)
    3. **Stratified Sampling**: Ensuring the test set maintains the same class distribution as the original data
    4. **Model Parameters**: Adjusted Random Forest parameters for better generalization:
       - Increased number of trees (200)
       - Increased max depth (15)
       - Added minimum samples requirements
    """)
    
    # Let users try the model with their own Pokemon
    st.header("Predict Legendary Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pokémon Stats")
        hp = st.slider("HP", 1, 255, 50)
        attack = st.slider("Attack", 1, 255, 50)
        defense = st.slider("Defense", 1, 255, 50)
        sp_attack = st.slider("Special Attack", 1, 255, 50)
        sp_defense = st.slider("Special Defense", 1, 255, 50)
        speed = st.slider("Speed", 1, 255, 50)
        height = st.slider("Height (m)", 0.1, 20.0, 1.0)
        weight = st.slider("Weight (kg)", 0.1, 1000.0, 50.0)
        generation = st.slider("Generation", 1, 8, 1)
    
    with col2:
        st.subheader("Pokémon Type")
        type1 = st.selectbox("Primary Type", sorted(df['type1'].unique()))
        type2 = st.selectbox("Secondary Type", ['None'] + sorted(df['type2'].dropna().unique()))
    
    # When the user clicks the button, make a prediction
    if st.button("Predict"):
        # Get all the Pokemon's stats
        pokemon_data = {
            'hp': hp,
            'attack': attack,
            'defense': defense,
            'sp_attack': sp_attack,
            'sp_defense': sp_defense,
            'speed': speed,
            'height_m': height,
            'weight_kg': weight,
            'generation': generation,
            'type1': type1,
            'type2': type2 if type2 != 'None' else None
        }
        
        # Get the probability that this Pokemon is legendary
        probability = ml_model.predict_legendary_probability(pokemon_data)
        
        # Show the results
        st.subheader("Prediction Result")
        st.write(f"Probability of being Legendary: {probability:.2%}")
        
        if probability > 0.5:
            st.success("This Pokémon is likely to be Legendary!")
        else:
            st.info("This Pokémon is likely to be non-Legendary.")