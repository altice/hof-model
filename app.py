import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load data for player lookup
df = pd.read_csv('data/model_data.csv')
people = pd.read_csv('data/People.csv')

# Merge names
df = df.merge(people[['playerID', 'nameFirst', 'nameLast']], on='playerID', how='left')
df['name'] = df['nameFirst'] + ' ' + df['nameLast']

# App
st.title("Is This Hall of Famer Legit?")
st.write("Enter a player name to see their Hall of Fame probability score.")

# Search
search = st.text_input("Player Name", placeholder="e.g. Barry Bonds")

if search:
    matches = df[df['name'].str.contains(search, case=False, na=False)]
    
    if matches.empty:
        st.warning("No player found. Try a different name.")
    else:
        player = matches.iloc[0]
        features = ['G', 'AB', 'R', 'H', 'HR', 'RBI', 'SB', 'BB', 'SO',
                    'seasons', 'BA', 'OBP', 'HR_rate', 'K_rate']
        X = player[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]
        inducted = player['inducted']

        st.subheader(f"{player['name']}")
        st.metric("HOF Probability", f"{prob:.1%}")
        
        if inducted == 1:
            st.success("✅ Inducted into the Hall of Fame")
        else:
            st.info("Not inducted")

        st.write("**Career Stats:**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("HR", int(player['HR']))
        col2.metric("BA", f"{player['BA']:.3f}")
        col3.metric("OBP", f"{player['OBP']:.3f}")
        col4.metric("Seasons", int(player['seasons']))