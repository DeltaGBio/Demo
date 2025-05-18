**Set the env**

```bash
conda env create --name mint --file=environment.yml
conda activate mint
git clone https://github.com/VarunUllanat/mint.git
pip install -e ./mint/
pip install streamlit
```

**Prepare the model**

Put the model checkpoints file in the `model` folder.

**Run the demo**

```bash
streamlit run app.py
```


