import joblib
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt  ## charting library - I use this for the alcohol sensitivity chart, since it lets me draw the line and the highlighted current-value dot as two layers on the same chart. Comes bundled with streamlit so no extra install needed.
from datetime import datetime  ## just need this for the timestamp column in the history table below
import base64  ## needed to embed the banner photo directly into the CSS - see get_base64_of_bin_file below

## Load the model I trained and saved in the notebook (Random Forest, tuned with RandomizedSearchCV). model.pkl needs to sit in this same folder.
model = joblib.load("wine_quality_model.pkl")

## page_icon shows up as the little tab icon in the browser
st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

## Streamlit reruns this whole script top to bottom on every interaction, and
## normally forgets everything each time. session_state is the way around
## that - I use it here to remember past predictions across reruns so users
## can compare a few wines in one sitting instead of losing the result every
## time they move a slider.
if "history" not in st.session_state:
    st.session_state.history = []

## Streamlit can't just take a plain file path in CSS (background: url("wine-bg.jpg")
## doesn't reliably resolve since the markdown gets rendered in its own iframe) -
## so I read the photo as bytes and base64-encode it, then embed that directly
## in the CSS as a data URI instead
def get_base64_of_bin_file(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img_base64 = get_base64_of_bin_file("wine-bg.jpg")

## Put the photo behind the WHOLE page (.stApp), not just one box. The dark
## gradient layered on top of it is there so text stays readable no matter
## how bright any part of the photo is - last time I only did this for one
## banner div and it looked fine, but doing it for the whole page means
## every bit of text in the main area now sits directly on top of the photo,
## so I also have to force all of it to a light colour here (the sidebar
## keeps its own separate light panel already, so it doesn't need this).
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, rgba(74,17,32,0.87), rgba(26,26,26,0.90)),
                    url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
    [data-testid="stMain"] p, [data-testid="stMain"] span, [data-testid="stMain"] label,
    [data-testid="stMarkdownContainer"] {{
        color: #f5e6e6 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍷 Wine Quality Predictor")
st.caption("Predicts a wine's quality score (0-10) from its lab measurements. "
           "Trained on 5,320 real red and white Vinho Verde wines.")

## All the inputs live in the sidebar so the main area is free for the
## result, the chart and the history table
st.sidebar.header("Enter Wine Lab Results")

wine_type_selected = st.sidebar.selectbox("Wine Type", ['red', 'white'])

## Slider min/max/default values are taken from the actual dataset stats
## (df.describe()) in my notebook, not guessed - so the defaults are roughly
## an "average" wine and users can't accidentally enter something outside
## what the model has ever seen
fixed_acidity_selected = st.sidebar.slider("Fixed Acidity (g/dm3)", 3.8, 15.9, 7.2)
volatile_acidity_selected = st.sidebar.slider("Volatile Acidity (g/dm3)", 0.08, 1.58, 0.34)
citric_acid_selected = st.sidebar.slider("Citric Acid (g/dm3)", 0.0, 1.66, 0.32)
residual_sugar_selected = st.sidebar.slider("Residual Sugar (g/dm3)", 0.6, 65.8, 5.4)
chlorides_selected = st.sidebar.slider("Chlorides (g/dm3)", 0.009, 0.611, 0.056)
free_sulfur_dioxide_selected = st.sidebar.slider("Free Sulfur Dioxide (mg/dm3)", 1.0, 289.0, 30.0)
total_sulfur_dioxide_selected = st.sidebar.slider("Total Sulfur Dioxide (mg/dm3)", 6.0, 440.0, 116.0)
density_selected = st.sidebar.slider("Density (g/cm3)", 0.98711, 1.03898, 0.99470, format="%.5f")
pH_selected = st.sidebar.slider("pH", 2.72, 4.01, 3.22)
sulphates_selected = st.sidebar.slider("Sulphates (g/dm3)", 0.22, 2.0, 0.53)
alcohol_selected = st.sidebar.slider("Alcohol (% vol)", 8.0, 14.9, 10.5)

predict_clicked = st.sidebar.button("Predict Wine Quality")


def build_input_row():
    """Turns the current sidebar values into a row the model can predict on.

    I made this a function instead of just writing it once because I need
    the exact same encode-and-reindex steps twice: once for the actual
    prediction, and again 25 times in a loop for the sensitivity chart below.
    """
    ## Convert current sidebar selections into a one-row DataFrame
    df_input = pd.DataFrame({
        'fixed acidity': [fixed_acidity_selected],
        'volatile acidity': [volatile_acidity_selected],
        'citric acid': [citric_acid_selected],
        'residual sugar': [residual_sugar_selected],
        'chlorides': [chlorides_selected],
        'free sulfur dioxide': [free_sulfur_dioxide_selected],
        'total sulfur dioxide': [total_sulfur_dioxide_selected],
        'density': [density_selected],
        'pH': [pH_selected],
        'sulphates': [sulphates_selected],
        'alcohol': [alcohol_selected],
        'wine_type': [wine_type_selected]
    })

    ## One-hot encode wine_type the same way I did in the notebook
    df_input = pd.get_dummies(df_input, columns=['wine_type'])

    ## get_dummies only makes a column for whichever type was picked (e.g.
    ## wine_type_red), but the model was trained with wine_type_white as the
    ## only dummy column. reindex lines the columns up with what the model
    ## expects and fills in 0 for anything missing - so picking "red" here
    ## correctly becomes wine_type_white = 0.
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
    return df_input


tab_predict, tab_about = st.tabs(["Prediction", "About"])

with tab_predict:
    if not predict_clicked and not st.session_state.history:
        st.info("Set the wine's lab measurements in the sidebar, then click **Predict Wine Quality**.")

    if predict_clicked:
        df_input = build_input_row()

        ## Predict
        y_unseen_pred = model.predict(df_input)[0]

        col_result, col_chart = st.columns([1, 2])

        with col_result:
            st.metric("Predicted Quality Score", f"{y_unseen_pred:.2f} / 10")

            ## Turning the raw number into a plain-English read, since a
            ## winemaker cares more about "is this batch okay" than the
            ## exact decimal - matches the notebook's business interpretation
            ## (flag anything predicted below 5)
            if y_unseen_pred >= 7:
                st.success("Predicted to be high quality.")
            elif y_unseen_pred >= 5:
                st.info("Predicted to be average quality.")
            else:
                st.warning("Predicted to be below average - may need review before bottling.")

        with col_chart:
            ## This chart is not from the notebook - I added it here to show
            ## how the prediction would change if only alcohol were
            ## different, keeping every other slider at what the user
            ## currently has. alcohol was the strongest feature in my
            ## feature importance chart, so it felt like the most useful one
            ## to visualise.
            alcohol_range = np.linspace(8.0, 14.9, 25)
            sweep_rows = [df_input.assign(alcohol=a) for a in alcohol_range]
            df_sweep = pd.concat(sweep_rows, ignore_index=True)
            sweep_preds = model.predict(df_sweep)

            df_chart = pd.DataFrame({
                'Alcohol (% vol)': alcohol_range,
                'Predicted Quality': sweep_preds
            })

            ## Plain line for the sweep, plus one highlighted dot for
            ## wherever the user's actual slider is sitting on that line
            line = alt.Chart(df_chart).mark_line(color="#c0392b").encode(
                x=alt.X('Alcohol (% vol)'),
                y=alt.Y('Predicted Quality', scale=alt.Scale(domain=[3, 9]))
            )
            current_point = alt.Chart(pd.DataFrame({
                'Alcohol (% vol)': [alcohol_selected],
                'Predicted Quality': [y_unseen_pred]
            })).mark_point(size=120, color="white", filled=True).encode(
                x='Alcohol (% vol)', y='Predicted Quality'
            )

            st.caption("How predicted quality responds to alcohol content (all other inputs held at your selected values)")
            st.altair_chart((line + current_point).properties(height=280), use_container_width=True)

        ## Save this prediction so it shows up in the history table below,
        ## even after the user changes sliders and predicts again
        st.session_state.history.append({
            'Time': datetime.now().strftime("%H:%M:%S"),
            'Type': wine_type_selected,
            'Alcohol': alcohol_selected,
            'Volatile Acidity': volatile_acidity_selected,
            'Predicted Quality': round(float(y_unseen_pred), 2)
        })

    ## Only show the history section once there's actually something to show
    if st.session_state.history:
        st.markdown("### Your Prediction History")
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        if st.button("Clear History"):
            st.session_state.history = []
            ## Without this the history list is cleared but the table on
            ## screen wouldn't update until the next time a widget is
            ## touched - st.rerun() forces the redraw straight away
            st.rerun()

with tab_about:
    st.markdown("""
    ### About this app
    This app predicts the **quality score (0-10)** of a wine from 11 physicochemical
    lab measurements plus wine type, using a **Random Forest Regressor** trained on
    5,320 real red and white *Vinho Verde* wines from the
    [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).

    **Model performance (test set):** MAE ~ 0.52 quality points, R^2 ~ 0.40 - on
    average, predictions land within about half a point of the expert tasting panel's score.

    **Strongest quality drivers:** alcohol content (positive) and volatile acidity (negative).
    """)
