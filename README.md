# Plant Disease Detection System

A deep learning project that can identify diseases in plant leaves just from a photo. You upload an image, and the model tells you what disease the plant has — or if it's healthy.

![Home Page](home_page.jpeg)

## What it does

The model is trained on around 87,000 leaf images across 38 different disease categories covering 14 crops like tomato, potato, corn, grape, apple and more. It uses a CNN trained with TensorFlow, and the interface is built with Streamlit so anyone can use it without knowing any code.

The app has three pages — a home page, an about section, and the main disease recognition page where you upload your image and hit predict.

## How to run it

Clone the repo and install the dependencies:

```bash
git clone https://github.com/mohammad70623/Plant-Disease-Detection-System-using-Deep-Learning.git
cd Plant-Disease-Detection-System-using-Deep-Learning
pip install streamlit tensorflow numpy pillow
```

Then place the trained model file at `../trained model/plant_desies_trained_model.keras` and run:

```bash
streamlit run main.py
```

## Model results

After 10 epochs the model hit around 98% training accuracy and 96.4% validation accuracy. The training history is saved in `training_hist.json` if you want to look at how it improved over time.

## Project files

- `main.py` — the Streamlit app
- `Train_Plant_Desies.ipynb` — training notebook
- `Test_plant_desies.ipynb` — testing and evaluation
- `training_hist.json` — accuracy and loss per epoch

## Built with

Python, TensorFlow, Streamlit, NumPy, Pillow
