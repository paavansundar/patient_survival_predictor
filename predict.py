

"""## Gradio Implementation"""
import gradio as gr
import gradio
import numpy as np
import joblib

save_file_name = "./patient_survival_predictor/trained_models/xgboost-model.pkl"
# Load your trained model

model = joblib.load(save_file_name)
#model.predict()

# Function for prediction
def predict_death_event(Age,Anemeia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,Sex,Smoker,time):

    prediction=""
    predict_arr=[]
    try:

      predict_arr.append(Age)
      predict_arr.append(1 if Anemeia == "Yes" else 0)
      predict_arr.append(creatinine_phosphokinase)
      predict_arr.append(1 if diabetes == "Yes" else 0)
      predict_arr.append(ejection_fraction)
      predict_arr.append(1 if high_blood_pressure == "Yes" else 0)
      predict_arr.append(platelets)
      predict_arr.append(serum_creatinine)
      predict_arr.append(serum_sodium)
      predict_arr.append(1 if Sex == "Male" else 0)
      predict_arr.append(1 if Smoker == "Yes" else 0)
      predict_arr.append(time)
      prediction="Will die" if model.predict(np.reshape(predict_arr,(1,12))) == 1 else "Will not die"
    except Exception as e:
      return e
    return prediction

"""For categorical user input, user [Radio](https://www.gradio.app/docs/radio) button component.

For numerical user input, user [Slider](https://www.gradio.app/docs/slider) component.
"""

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = [gr.Slider(1, 100),gr.Radio(["Yes","No"]),gr.Slider(1000, 10000),gr.Radio(["Yes","No"]),gr.Slider(1, 100),gr.Radio(["Yes","No"]),gr.Slider(100000, 1000000),gr.Slider(1, 10),gr.Slider(100, 10000),gr.Radio(["Male","Female"]),gr.Radio(["Yes","No"]),gr.Slider(1, 12)],
                         outputs = "label",
                         title = title,
                         description = description)

iface.launch(share = True)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface

