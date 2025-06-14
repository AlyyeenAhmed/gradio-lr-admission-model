import joblib
import gradio as gr

# Load the saved model
model = joblib.load("linear_model2.joblib")

def predictp(GREscore, TOEFLscore, UniversityRating, SOP, LOR, CGPA, Research):
    input_data = [[GREscore, TOEFLscore, UniversityRating, SOP, LOR, CGPA, Research]]
    result = model.predict(input_data)[0]
    result = max(0, min(result, 1))
    return f"Your probability to get admission at university is: {result * 100:.2f} %"

demo = gr.Interface(
    fn=predictp,
    inputs=[
        gr.Slider(260, 340, label="GRE Score"),
        gr.Slider(90, 120, label="TOEFL Score"),
        gr.Slider(1, 5, step=1, label="University Rating"),
        gr.Slider(1.0, 5.0, label="SOP Strength"),
        gr.Slider(1.0, 5.0, label="LOR Strength"),
        gr.Slider(6.0, 10.0, label="CGPA"),
        gr.Radio([0, 1], label="Research Experience"),
    ],
    outputs="text",
    title="University Admission Predictor",
)

demo.launch()
