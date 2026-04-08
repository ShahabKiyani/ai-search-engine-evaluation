# 1. Clone repo
git clone https://github.com/ShahabKiyani/ai-search-engine-evaluation.git
cd ai-search-engine-evaluation

# 2. Create virtual environment in terminal
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies (run in terminal after you see (venv) in terminal bar)
pip install -r requirements.txt


from beir import util
util.download_and_unzip("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip", "datasets")

run:
python build_nq_index.py


run the app:
streamlit run main2_streamlit.py
